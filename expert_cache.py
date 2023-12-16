import random
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Sequence, Dict, Optional
from collections import deque, defaultdict

import torch
import torch.nn as nn

ExpertUID = Any


@dataclass(frozen=False)
class ExpertInfo:
    uid: ExpertUID
    eviction_group: int
    offloaded: bool
    index: int
    last_use: int


class ExpertCache:
    def __init__(self, make_module: callable, main_size: int, offload_size: int, buffer_size: int):
        """Dynamically loads an array of modules with identical hyperparameters"""
        
        self.counter = 0
        
        # group_stats: eviction_group -> dict {'hits', 'misses'}
        self.group_stats: Dict[int, Dict[str, int]] = defaultdict(lambda: {'hits': 0, 'misses': 0})

        self.registered_experts: Dict[ExpertUID, ExpertInfo] = dict()
        
        self.main_modules = []
        for i in range(main_size):
            module = make_module()
            assert isinstance(module.storage, torch.UntypedStorage)
            self.main_modules.append(module)
            if i == 0:
                self.module_type = type(module)
                self.module_size = len(module.storage)
                self.device = device = module.storage.device
            else:
                assert isinstance(module, self.module_type)
                assert len(module.storage) == self.module_size
                assert module.storage.device == self.device
                assert module.storage is not self.main_modules[0].storage
        
        self.main_infos: Sequence[Optional[ExpertInfo]] = [None for _ in range(main_size)]

        self.offloaded_storages = [module.storage.cpu().clone().pin_memory(device) for _ in range(offload_size)]
        self.offloaded_infos: Sequence[Optional[ExpertInfo]] = [None for _ in range(offload_size)]
        
        # eviction_group -> list of infos
        self.group_infos: Dict[int, List[ExpertInfo]] = defaultdict(list)

        # temporary storage to shave off latency
        self.device_expert_buffers = deque([deepcopy(module).to(device) for _ in range(buffer_size)])
        self.offloaded_storage_buffers = deque([module.storage.cpu().clone().pin_memory(device) for _ in range(buffer_size)])

    def add_expert(self, uid: ExpertUID, module: MixtralExpertWrapper, eviction_group: int = 0,
                   offload: Optional[bool] = None):
        """Register an expert to the cache and associate it with uid"""
        assert isinstance(module, self.module_type)
        return self.add_expert_storage(uid, module.storage, eviction_group=eviction_group, offload=offload)

    def add_expert_storage(self, uid: ExpertUID, storage: torch.UntypedStorage,
                           eviction_group: int = 0, offload: Optional[bool] = None):
        assert uid not in self.registered_experts, f"expert {uid} already registered"
        assert isinstance(storage, torch.UntypedStorage)
        assert len(storage) == self.module_size
        
        self.counter += 1
        if offload is None or not offload: # False or None
            for i in range(len(self.main_modules)):
                if self.main_infos[i] is None:
                    self.main_modules[i].storage.copy_(storage)
                    info = ExpertInfo(
                        uid, eviction_group=eviction_group, offloaded=False, index=i, last_use=self.counter
                    )
                    self.registered_experts[uid] = self.main_infos[i] = info
                    self.group_infos[eviction_group].append(info)
                    
                    return  # done allocating; found spot on device
        if offload is None or offload:  # True or None
            for i in range(len(self.offloaded_storages)):
                if self.offloaded_infos[i] is None:
                    self.offloaded_storages[i].copy_(storage)
                    info = ExpertInfo(
                        uid, eviction_group=eviction_group, offloaded=True, index=i, last_use=self.counter
                    )
                    self.registered_experts[uid] = self.offloaded_infos[i] = info
                    self.group_infos[eviction_group].append(info)
                    
                    return  # done allocating; found an offloaded spot
        raise ValueError("Cache is full")

    def choose_evicted_experts(self, to_load: Sequence[ExpertInfo]) -> Sequence[ExpertInfo]:
        """Choose on-device experts to be replaced when loading the specified infos. Cannot choose one of to_load infos"""
        
        to_load_by_group: Dict[int, List[int]] = defaultdict(list)
        to_evict: List[Optional[ExpertInfo]] = [None] * len(to_load)
        
        for i, info in enumerate(to_load):
            assert info.offloaded
            to_load_by_group[info.eviction_group].append(i)
        
        # Max Ryabinin and Algo God, forgive me
        
        for group, indices in to_load_by_group.items():
            self.group_infos[group] = sorted(self.group_infos[group], key=lambda info: info.last_use)
            
            idx = 0
            
            for info in self.group_infos[group]:
                if info.offloaded:
                    continue
                
                to_evict[indices[idx]] = info
                
                idx += 1
                if idx >= len(indices):
                    break
            else:
                assert False, f"Not enough gpu slots in group {eviction_group}"
        
        for info_to_load, info_to_evict in zip(to_load, to_evict):
            assert info_to_load.eviction_group == info_to_evict.eviction_group
            assert not info_to_evict.offloaded
        
        return to_evict

    def load_experts(self, *uids: ExpertUID) -> Sequence[MixtralExpertWrapper]:
        assert len(set(uids)) == len(uids)
        assert len(uids) <= len(self.main_modules)
        assert len(uids) <= len(self.device_expert_buffers)  # can be lifted at the cost of slower memory
        infos = [self.registered_experts[uid] for uid in uids]
        
        for info in infos:
            self.counter += 1
            info.last_use = self.counter
            self.group_stats[info.eviction_group]['hits'] += 1

        infos_to_load = [info for info in infos if info.offloaded]
        infos_to_evict = self.choose_evicted_experts(infos_to_load)
        
        assert not any(info.offloaded for info in infos_to_evict)
        assert len(infos_to_evict) == len(infos_to_load)

        for i in range(len(infos_to_load)):
            load_info = infos_to_load[i]
            evict_info = infos_to_evict[i]
            
            self._swap_infos(load_info, evict_info)

        assert not any(info.offloaded for info in infos)
        return [self.main_modules[info.index] for info in infos]
    
    def _swap_infos(self, load_info: ExpertInfo, evict_info: ExpertInfo) -> None:
        if load_info == evict_info:
            return
        
        assert load_info.eviction_group == evict_info.eviction_group
        
        self.group_stats[evict_info.eviction_group]['misses'] += 1
        
        # swap a single on-device expert with a single offloaded expert using buffers for parallelism
        offloaded_storage_buffer = self.offloaded_storage_buffers.popleft()
        device_expert_buffer = self.device_expert_buffers.popleft()
        device_expert_buffer.storage.copy_(self.offloaded_storages[load_info.index], non_blocking=True)
        offloaded_storage_buffer.copy_(self.main_modules[evict_info.index].storage, non_blocking=True)

        self.device_expert_buffers.append(self.main_modules[evict_info.index])
        self.main_modules[evict_info.index] = device_expert_buffer
        self.offloaded_storage_buffers.append(self.offloaded_storages[load_info.index])
        self.offloaded_storages[load_info.index] = offloaded_storage_buffer

        self.main_infos[evict_info.index] = load_info
        self.offloaded_infos[load_info.index] = evict_info
        evict_info.offloaded, load_info.offloaded = load_info.offloaded, evict_info.offloaded
        evict_info.index, load_info.index = load_info.index, evict_info.index
