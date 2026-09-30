## helpers for optional multi-gpu training with torchrun and DDP

from __future__ import annotations
import os
from dataclasses import dataclass
import torch
import torch.distributed as dist

@dataclass(frozen=True)
class DistributedContext:
    rank: int = 0 ## which copy
    local_rank: int = 0 ## which gpu
    world_size: int = 1 ## total?

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool: ## the orchestrator
        return self.rank == 0

def setup_distributed() -> DistributedContext:
    ## reads numbers from torchrun, picks the right device, connects/orchestrate
    
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size == 1:
        return DistributedContext()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    return DistributedContext(rank=rank, local_rank=local_rank, world_size=world_size)

def get_context() -> DistributedContext:
    ## contextual info
    if not (dist.is_available() and dist.is_initialized()):
        return DistributedContext()
    return DistributedContext(
        rank=dist.get_rank(),
        local_rank=int(os.environ.get("LOCAL_RANK", "0")),
        world_size=dist.get_world_size(),
    )

def cleanup_distributed() -> None:
    ## disconnect
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

def sum_across_processes(value: float) -> float:
    ## reduction through sum op
    context = get_context()
    if not context.enabled:
        return value
    total = torch.tensor(
        float(value), dtype=torch.float64, device=torch.device("cuda", context.local_rank)
    )
    dist.all_reduce(total, op=dist.ReduceOp.SUM)
    return float(total.item())

def gather_across_processes(tensors: list[torch.Tensor]) -> list[torch.Tensor]:
    ## collect all device tensors for epoch level info
    context = get_context()
    if not context.enabled:
        return tensors
    gathered: list[list[torch.Tensor] | None] = [None] * context.world_size
    dist.all_gather_object(gathered, tensors)
    return [tensor for part in gathered for tensor in part if part is not None]