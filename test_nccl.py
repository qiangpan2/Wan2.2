import os
import torch
import torch.distributed as dist

def setup_distributed():
    """初始化分布式环境"""
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
    )
    rank = dist.get_rank()
    torch.cuda.set_device(rank)  # 绑定当前GPU
    return rank

def test_broadcast(base_seed):
    """测试广播一个对象列表"""
    rank = dist.get_rank()
    if rank == 0:
        print(f"[Rank {rank}] Broadcasting seed: {base_seed}")

    dist.broadcast_object_list(base_seed, src=0)

    print(f"[Rank {rank}] Received seed: {base_seed}")
    return base_seed[0]

def main():
    rank = setup_distributed()
    # 定义base_seed（主rank生成，副rank接收）
    if rank == 0:
        base_seed = [42]  # 可以是任意可序列化的对象（如int, str, dict）
    else:
        base_seed = [-1]

    seed = test_broadcast(base_seed)

    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()