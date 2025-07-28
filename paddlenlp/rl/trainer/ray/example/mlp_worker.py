import os
import ray
import paddle
import paddle.nn as nn
import paddle.distributed as dist
from paddle.distributed import fleet
from paddle.distributed.fleet import meta_parallel
from paddle.distributed.fleet.meta_parallel import LayerDesc, PipelineLayer
from paddle.distributed.sharding import group_sharded_parallel

class ToyModel(nn.Layer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)

class ToyTPModel(nn.Layer):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = meta_parallel.ColumnParallelLinear(input_size, hidden_size, gather_output=False)
        self.linear2 = meta_parallel.RowParallelLinear(hidden_size, output_size, input_is_parallel=True)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x

class ToyPPModel(PipelineLayer):
    def __init__(self, input_size, hidden_size, output_size, **kwargs):
        layers = [
            LayerDesc(nn.Linear, input_size, hidden_size),
            LayerDesc(nn.ReLU),
            LayerDesc(nn.Linear, hidden_size, output_size),
        ]
        super().__init__(layers=layers, loss_fn=nn.MSELoss(), **kwargs)

class ToyTPandPPModel(PipelineLayer):
    def __init__(self, input_size, hidden_size, output_size, **kwargs):
        layers = [
            LayerDesc(meta_parallel.ColumnParallelLinear, input_size, hidden_size, gather_output=True),
            LayerDesc(nn.ReLU),
            LayerDesc(meta_parallel.RowParallelLinear, hidden_size, output_size, input_is_parallel=False),
        ]
        super().__init__(layers=layers, loss_fn=nn.MSELoss(), **kwargs)

@ray.remote(num_gpus=1)
class MLPWorker:
    def __init__(self, env_vars):
        os.environ.update(env_vars)

    def train(self):
        # Step 1: 配置所有超参数和结构参数
        dp_degree = 2
        tp_degree = 2
        pp_degree = 1
        sp_degree = 2

        input_size = 10
        hidden_size = 10
        output_size = 1

        micro_batch_size = 16
        accumulate_steps = 1
        num_steps = 10
        learning_rate = 1e-3

        strategy = fleet.DistributedStrategy()
        strategy.hybrid_configs = {
            "dp_degree": dp_degree,
            "mp_degree": tp_degree,
            "pp_degree": pp_degree,
            "sharding_degree": sp_degree,
            "sep_degree": 1,
        }
        strategy.pipeline_configs = {
            "accumulate_steps": accumulate_steps,
            "micro_batch_size": micro_batch_size,
        }
        fleet.init(is_collective=True, strategy=strategy)

        hcg = fleet.get_hybrid_communicate_group()
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        dp_id = hcg.get_data_parallel_rank()
        pp_id = hcg.get_stage_id()
        tp_id = hcg.get_model_parallel_rank()

        print(f"[Rank {rank}/{world_size}] Starting training...")

        # Step 2: Build model and optimizer
        # 任选一种模型结构
        model = ToyTPModel(input_size, hidden_size, output_size)
        # model = ToyPPModel(input_size, hidden_size, output_size, num_stages=pp_degree, topology=hcg._topo)
        # model = ToyTPandPPModel(input_size, hidden_size, output_size, num_stages=pp_degree, topology=hcg._topo)

        model = fleet.distributed_model(model)
        optimizer = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=learning_rate)
        model, optimizer, _ = group_sharded_parallel(model, optimizer, level="p_g_os")
        optimizer = fleet.distributed_optimizer(optimizer)

        # Step 3: Fake data
        x = paddle.randn([micro_batch_size, input_size])
        y = paddle.randn([micro_batch_size, output_size])

        # Step 4: Train
        for step in range(num_steps):
            pred = model(x)
            loss = paddle.nn.functional.mse_loss(pred, y)
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            print(f"[Rank {rank}] Step {step}, Loss = {loss.numpy()}")

if __name__ == "__main__":
    pass