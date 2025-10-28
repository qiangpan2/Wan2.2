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


def test_broadcast_tensor():
    rank = dist.get_rank()
    if rank == 0:
        seed_tensor = torch.tensor([42], device=f"cuda:{rank}")
    else:
        seed_tensor = torch.zeros(1, device=f"cuda:{rank}")
    dist.broadcast(seed_tensor, src=0)
    print(f"[Rank {rank}] Received seed: {seed_tensor.item()}")


def test_broadcast(base_seed):
    """测试广播一个对象列表"""
    rank = dist.get_rank()
    if rank == 0:
        print(f"[Rank {rank}] Broadcasting seed: {base_seed}")
    else:
        base_seed = [None]  # 非主rank初始化空列表

    # 广播数据
    dist.broadcast_object_list(base_seed, src=0)

    # 验证结果
    print(f"[Rank {rank}] Received seed: {base_seed}")
    return base_seed[0]

def main():
    
    is_tensor=True
    rank = setup_distributed()
    # 定义base_seed（主rank生成，副rank接收）
    if rank == 0:
        base_seed = [42]  # 可以是任意可序列化的对象（如int, str, dict）
    else:
        base_seed = [None]

    if is_tensor:
        test_broadcast_tensor()
    else
        seed = test_broadcast(base_seed)

    # 确保所有卡完成同步
    dist.barrier()

    # 清理进程组
    dist.destroy_process_group()

if __name__ == "__main__":
    main()