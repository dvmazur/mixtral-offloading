from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Iterator, Tuple, List
from collections import deque, defaultdict, OrderedDict
from .expert_wrapper import MixtralExpertWrapper

import torch
from torch import nn

ExpertUID = Any


@dataclass(frozen=False)
class ExpertInfo:
    uid: ExpertUID
    eviction_group: int
    offloaded: bool
    index: int


@dataclass
class EvictionGroupInfo:
    # infos in main and offload devices; ordered from least recently used to most
    main_infos: OrderedDict[ExpertUID, ExpertInfo] = field(default_factory=OrderedDict)
    offloaded_infos: OrderedDict[ExpertUID, ExpertInfo] = field(default_factory=OrderedDict)
    hits: int = field(default=0)
    misses: int = field(default=0)

    def add(self, info: ExpertInfo):
        infos_odict = self.offloaded_infos if info.offloaded else self.main_infos
        assert info.uid not in infos_odict, f"expert {info.uid} already exists"
        infos_odict[info.uid] = info

    def choose_expert_to_evict(self) -> ExpertInfo:
        for uid, info in self.main_infos.items():
            return info  # least recently used
        raise ValueError("No evictable experts")

    def swap(self, info_to_load: ExpertInfo, info_to_evict: ExpertInfo):
        assert info_to_load.uid in self.offloaded_infos and info_to_evict.uid in self.main_infos
        self.main_infos[info_to_load.uid] = self.offloaded_infos.pop(info_to_load.uid)
        self.main_infos.move_to_end(info_to_load.uid, last=True)
        self.offloaded_infos[info_to_evict.uid] = self.main_infos.pop(info_to_evict.uid)

    def mark_used(self, info: ExpertInfo):
        if info.uid in self.main_infos:
            self.main_infos.move_to_end(info.uid, last=True)
            self.hits += 1
        elif info.uid in self.offloaded_infos:
            self.offloaded_infos.move_to_end(info.uid, last=True)
            self.misses += 1
        else:
            raise ValueError(f"Expert {info} not in group")


class ExpertCache:
    def __init__(self, make_module: callable, main_size: int, offload_size: int, buffer_size: int):
        """Dynamically loads an array of modules with identical hyperparameters"""
        self.module_type = self.module_size = self.device = None
        self.active = False

        self.registered_experts: Dict[ExpertUID, ExpertInfo] = dict()

        self.main_modules = [self._check_module(make_module()) for i in range(main_size)]
        self.main_infos: List[Optional[ExpertInfo]] = [None for _ in range(main_size)]

        assert self.module_size is not None
        self.offloaded_storages = [
            torch.UntypedStorage(self.module_size).pin_memory(self.device) for _ in range(offload_size)]
        self.offloaded_infos: List[Optional[ExpertInfo]] = [None for _ in range(offload_size)]

        # temporary storage to shave off latency
        self.device_expert_buffers = deque([self._check_module(make_module()) for _ in range(buffer_size)])
        self.offloaded_storage_buffers = deque([
            torch.UntypedStorage(self.module_size).pin_memory(self.device) for _ in range(buffer_size)])
        self.group_infos: Dict[int, EvictionGroupInfo] = defaultdict(EvictionGroupInfo)

    def _check_module(self, module: MixtralExpertWrapper):
        assert isinstance(module.storage, torch.UntypedStorage)
        if self.module_type is None:
            self.module_type = type(module)
            self.module_size = len(module.storage)
            self.device = module.storage.device
        else:
            assert isinstance(module, self.module_type)
            assert len(module.storage) == self.module_size
            assert module.storage.device == self.device
        return module

    def add_expert(self, uid: ExpertUID, module: MixtralExpertWrapper, eviction_group: int = 0,
                   offload: Optional[bool] = None):
        """Register an expert to the cache and associate it with uid"""
        assert self.module_type is not None
        assert isinstance(module, self.module_type)
        return self.add_expert_storage(uid, module.storage, eviction_group=eviction_group, offload=offload)

    def add_expert_storage(self, uid: ExpertUID, storage: torch.UntypedStorage,
                           eviction_group: int = 0, offload: Optional[bool] = None):
        assert uid not in self.registered_experts, f"expert {uid} already registered"
        assert isinstance(storage, torch.UntypedStorage)
        assert len(storage) == self.module_size

        if offload is None or not offload:  # False or None
            for i in range(len(self.main_modules)):
                if self.main_infos[i] is None:
                    self.main_modules[i].storage.copy_(storage)
                    info = ExpertInfo(uid, eviction_group=eviction_group, offloaded=False, index=i)
                    self.registered_experts[uid] = self.main_infos[i] = info
                    self.group_infos[eviction_group].add(info)
                    return  # done allocating; found spot on device
        if offload is None or offload:  # True or None
            for i in range(len(self.offloaded_storages)):
                if self.offloaded_infos[i] is None:
                    self.offloaded_storages[i].copy_(storage)
                    info = ExpertInfo(uid, eviction_group=eviction_group, offloaded=True, index=i)
                    self.registered_experts[uid] = self.offloaded_infos[i] = info
                    self.group_infos[eviction_group].add(info)
                    return  # done allocating; found an offloaded spot
        raise ValueError("Cache is full")

    def load_experts(
            self, *uids: ExpertUID, unordered: bool = False) -> Iterator[Tuple[ExpertUID, MixtralExpertWrapper]]:
        """
        :example:
        >>> for uid, expert in expert_cache.load_experts(*list_of_uids, unordered=True):
        >>>     for uid, expert in expert_iter:
        >>>         result += expert(x) * get_moe_weight(uid)

        :param uids: iterate over the specified expert uids. Same uids as in add_expert
        :param unordered: if True, allows cache to iterate experts in arbitrary order
            The order is chosen to minimize the total wait time.
        :returns: an iterator that yields (uid, expert) pairs, only usable inside the for loop

        """
        assert len(set(uids)) == len(uids)
        assert not self.active, "already loading experts; buffers are busy"
        if unordered:  # yield non-offloaded experts first
            uids = sorted(uids, key=lambda uid: self.registered_experts[uid].offloaded)
        infos = [self.registered_experts[uid] for uid in uids]

        assert len(set(info.eviction_group for info in infos)) == 1, "experts must be in the same evicton group"
        eviction_group = self.group_infos[infos[0].eviction_group]
        for info in infos:
            eviction_group.mark_used(info)

        try:
            self.active = True
            # save pre-loaded experts before they can be swapped
            pre_loaded_infos = deque([info for info in infos if not info.offloaded])
            pre_loaded_experts = deque([self.main_modules[info.index] for info in pre_loaded_infos])

            # begin loading experts into free buffers in background (via non-blocking copy)
            infos_to_load = deque([info for info in infos if info.offloaded])
            infos_in_loading = deque([])
            experts_in_loading = deque([])
            window_size = min(len(self.device_expert_buffers) - 1,
                              len(eviction_group.main_infos),
                              len(infos_to_load))
            for _ in range(window_size):
                info_to_load = infos_to_load.popleft()
                infos_in_loading.append(info_to_load)
                experts_in_loading.append(
                    self._swap(info_to_load, eviction_group.choose_expert_to_evict()))

            for info in infos:
                if len(pre_loaded_infos) > 0 and info is pre_loaded_infos[0]:
                    pre_loaded_infos.popleft()
                    yield (info.uid, pre_loaded_experts.popleft())
                elif len(infos_in_loading) > 0 and info is infos_in_loading[0]:
                    infos_in_loading.popleft()
                    yield (info.uid, experts_in_loading.popleft())
                    if len(infos_to_load) > 0:
                        info_to_load = infos_to_load.popleft()
                        infos_in_loading.append(info_to_load)
                        experts_in_loading.append(
                            self._swap(info_to_load, eviction_group.choose_expert_to_evict()))
                else:
                    raise RuntimeError("internal error: caching algorithm failed")
        finally:
            self.active = False

    def _swap(self, info_to_load: ExpertInfo, info_to_evict: ExpertInfo) -> nn.Module:
        """Swap an offloaded expert (info_to_load) with an on-device expert (info_to_evict) return the loaded expert"""
        assert info_to_load.offloaded and not info_to_evict.offloaded
        assert info_to_load.eviction_group == info_to_evict.eviction_group
        # swap a single on-device expert with a single offloaded expert using buffers for parallelism
        offloaded_storage_buffer = self.offloaded_storage_buffers.popleft()
        device_expert_buffer = self.device_expert_buffers.popleft()
        device_expert_buffer.storage.copy_(self.offloaded_storages[info_to_load.index], non_blocking=True)
        offloaded_storage_buffer.copy_(self.main_modules[info_to_evict.index].storage, non_blocking=True)

        self.device_expert_buffers.append(self.main_modules[info_to_evict.index])
        self.main_modules[info_to_evict.index] = device_expert_buffer
        self.offloaded_storage_buffers.append(self.offloaded_storages[info_to_load.index])
        self.offloaded_storages[info_to_load.index] = offloaded_storage_buffer

        self.main_infos[info_to_evict.index] = info_to_load
        self.offloaded_infos[info_to_load.index] = info_to_evict
        info_to_evict.offloaded, info_to_load.offloaded = info_to_load.offloaded, info_to_evict.offloaded
        info_to_evict.index, info_to_load.index = info_to_load.index, info_to_evict.index
        self.group_infos[info_to_load.eviction_group].swap(info_to_load, info_to_evict)
        return device_expert_buffer
