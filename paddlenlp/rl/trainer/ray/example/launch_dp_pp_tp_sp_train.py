import ray
import os
from mlp_worker import MLPWorker

if __name__ == "__main__":
    ray.init()
    world_size = 8
    base_port = 6170
    ip = "127.0.0.1"
    endpoints = [f"{ip}:{base_port+i}" for i in range(world_size)]
    trainer_endpoints = ",".join(endpoints)
    print(f"{endpoints=}")
    print(f"{trainer_endpoints=}")

    workers = []
    for rank in range(world_size):
        env_vars = {
            "PADDLE_TRAINER_ID": str(rank),
            "PADDLE_TRAINERS_NUM": str(world_size),
            "PADDLE_CURRENT_ENDPOINT": endpoints[rank],
            "PADDLE_TRAINER_ENDPOINTS": trainer_endpoints,
            "FLAGS_selected_gpus": "0",  # 只用可见的那张卡
            # 不要设置 CUDA_VISIBLE_DEVICES
        }
        worker = MLPWorker.options(runtime_env={"env_vars": env_vars}).remote(env_vars)
        workers.append(worker)

    ray.get([worker.train.remote() for worker in workers])