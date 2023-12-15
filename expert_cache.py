import random
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Sequence, Dict, Optional
from collections import deque

import torch
import torch.nn as nn


class ModuleWithStorage(nn.Sequential):  # dummy
    def __init__(self, *args, size: int):
        super().__init__(*args)
        self.storage = torch.UntypedStorage(size)


ExpertUID = Any


@dataclass(frozen=False)
class ExpertInfo:
    uid: ExpertUID
    offloaded: bool
    index: int
    last_use: int


class ExpertCache:
    def __init__(self, module: ModuleWithStorage,
                 device: torch.device, main_size: int, offload_size: int, buffer_size: int):
        """Dynamically loads an array of modules with identical hyperparameters"""
        assert isinstance(module, ModuleWithStorage)
        assert isinstance(module.storage, torch.UntypedStorage)
        self.device = device
        self.module_type = type(module)
        self.module_size = len(module.storage)
        self.counter = 0

        self.registered_experts: Dict[ExpertUID, ExpertInfo] = dict()

        self.main_modules = [deepcopy(module).to(device) for _ in range(main_size)]
        self.main_infos: Sequence[Optional[ExpertInfo]] = [None for _ in range(main_size)]

        self.offloaded_storages = [module.storage.cpu().clone().pin_memory(device) for _ in range(offload_size)]
        self.offloaded_infos: Sequence[Optional[ExpertInfo]] = [None for _ in range(offload_size)]

        # temporary storage to shave off latency
        self.device_expert_buffers = deque([deepcopy(module).to(device) for _ in range(buffer_size)])
        self.offloaded_storage_buffers = deque([module.storage.cpu().clone().pin_memory(device) for _ in range(buffer_size)])

    def add_expert(self, uid: ExpertUID, module: ModuleWithStorage):
        """Register an expert to the cache and associate it with uid"""
        assert isinstance(module, self.module_type)
        return self.add_expert_storage(uid, module.storage)

    def add_expert_storage(self, uid: ExpertUID, storage: torch.UntypedStorage):
        assert uid not in self.registered_experts, f"expert {uid} already registered"
        assert isinstance(storage, torch.UntypedStorage)
        assert len(storage) == self.module_size
        self.counter += 1
        for i in range(len(self.main_modules)):
            if self.main_infos[i] is None:
                self.main_modules[i].storage.copy_(storage)
                self.registered_experts[uid] = self.main_infos[i] = ExpertInfo(
                    uid, offloaded=False, index=i, last_use=self.counter)
                return  # done allocating; found spot on device
        for i in range(len(self.offloaded_storages)):
            if self.offloaded_infos[i] is None:
                self.offloaded_storages[i].copy_(storage)
                self.registered_experts[uid] = self.offloaded_infos[i] = ExpertInfo(
                    uid, offloaded=True, index=i, last_use=self.counter)
                return  # done allocating; found an offloaded spot
        raise ValueError("Cache is full")

    def choose_evicted_experts(self, to_load: Sequence[ExpertInfo]) -> Sequence[ExpertInfo]:
        """Choose on-device experts to be replaced when loading the specified infos. Cannot choose one of to_load infos"""
        num_replaced = sum(info.offloaded for info in to_load)
        available_infos = [info for info in self.main_infos if info not in to_load]
        return random.sample(available_infos, k=num_replaced)  # TODO unfuck

    def load_experts(self, *uids: ExpertUID) -> Sequence[ModuleWithStorage]:
        assert len(set(uids)) == len(uids)
        assert len(uids) <= len(self.main_modules)
        assert len(uids) <= len(self.device_expert_buffers)  # can be lifted at the cost of slower memory
        infos = [self.registered_experts[uid] for uid in uids]
        for info in infos:
            self.counter += 1
            info.last_use = self.counter

        infos_to_load = [info for info in infos if info.offloaded]
        infos_to_evict = self.choose_evicted_experts(infos)
        assert not any(info.offloaded for info in infos_to_evict)
        assert len(infos_to_evict) == len(infos_to_load)

        for i in range(len(infos_to_load)):
            # swap a single on-device expert with a single offloaded expert using buffers for parallelism
            offloaded_storage_buffer = self.offloaded_storage_buffers.popleft()
            device_expert_buffer = self.device_expert_buffers.popleft()
            device_expert_buffer.storage.copy_(self.offloaded_storages[infos_to_load[i].index], non_blocking=True)
            offloaded_storage_buffer.copy_(self.main_modules[infos_to_evict[i].index].storage, non_blocking=True)

            self.device_expert_buffers.append(self.main_modules[infos_to_evict[i].index])
            self.main_modules[infos_to_evict[i].index] = device_expert_buffer
            self.offloaded_storage_buffers.append(self.offloaded_storages[infos_to_load[i].index])
            self.offloaded_storages[infos_to_load[i].index] = offloaded_storage_buffer

            self.main_infos[infos_to_evict[i].index] = infos_to_load[i]
            self.offloaded_infos[infos_to_load[i].index] = infos_to_evict[i]
            infos_to_evict[i].offloaded, infos_to_load[i].offloaded = infos_to_load[i].offloaded, infos_to_evict[i].offloaded
            infos_to_evict[i].index, infos_to_load[i].index = infos_to_load[i].index, infos_to_evict[i].index


        assert not any(info.offloaded for info in infos)
        return [self.main_modules[info.index] for info in infos]
