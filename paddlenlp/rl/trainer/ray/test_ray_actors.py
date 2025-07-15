import ray
import paddle
from types import SimpleNamespace
from paddlenlp.rl.trainer.ray.policy_actor import PolicyModelRayActor
from paddlenlp.rl.trainer.ray.critic_actor import CriticModelRayActor
from paddlenlp.rl.trainer.ray.reference_actor import ReferenceModelRayActor
from paddlenlp.rl.trainer.ray.reward_actor import RewardModelRayActor
from paddlenlp.rl.trainer.ray.launcher import RayActorGroup

def get_args_for_test():
    # 你可以参考 run_rl.py 里的参数构造方式
    # 这里用 SimpleNamespace 方便模拟 argparse/HfArgumentParser 的结果
    model_args = SimpleNamespace(model_name_or_path="Qwen/Qwen2.5-0.5B-Instruct")
    data_args = SimpleNamespace(max_seq_length=128)
    training_args = SimpleNamespace(per_device_train_batch_size=4, num_train_epochs=1)
    common_config = SimpleNamespace(some_config=1)
    reshard_controller = None
    trainer_args = {}  # 其他可选参数
    return model_args, data_args, training_args, common_config, reshard_controller, trainer_args

def main():
    ray.init()
    strategy_config = {
        "dp_degree": 2,
        "mp_degree": 1,
        "sharding_degree": 1,
        "pp_degree": 1,
        "temperature": 1.0,
        "use_fp32_compute": False,
        "per_device_logprob_batch_size": 1,
        "max_new_tokens": 512,
        "top_p": 1.0,
        "do_sample": True,
    }
    num_nodes = 1
    num_gpus_per_node = 2

    # 构造参数
    model_args, data_args, training_args, common_config, reshard_controller, trainer_args = get_args_for_test()

    # 创建 Actor Groups
    reference_model_group = RayActorGroup(
        num_nodes=num_nodes,
        num_gpus_per_node=num_gpus_per_node,
        ray_actor_type=ReferenceModelRayActor,
        num_gpus_per_actor=1,
    )
    actor_model_group = RayActorGroup(
        num_nodes=num_nodes,
        num_gpus_per_node=num_gpus_per_node,
        ray_actor_type=PolicyModelRayActor,
        num_gpus_per_actor=1,
    )
    reward_model_group = RayActorGroup(
        num_nodes=num_nodes,
        num_gpus_per_node=num_gpus_per_node,
        ray_actor_type=RewardModelRayActor,
        num_gpus_per_actor=1,
    )

    print("初始化 Reference Model...")
    ray.get(reference_model_group.async_init_model_from_pretrained(
        strategy_config, model_args, data_args, training_args, common_config, reshard_controller, **trainer_args
    ))

    print("初始化 Actor Model...")
    ray.get(actor_model_group.async_init_model_from_pretrained(
        strategy_config, model_args, data_args, training_args, common_config, reshard_controller, **trainer_args
    ))

    print("初始化 Reward Model...")
    ray.get(reward_model_group.async_init_model_from_pretrained(
        strategy_config, model_args, data_args, training_args, common_config, reshard_controller, **trainer_args
    ))

    # 准备测试数据
    batch_size = 4
    seq_len = 128
    input_ids = paddle.randint(0, 1000, (batch_size, seq_len))
    attention_mask = paddle.ones((batch_size, seq_len))
    position_ids = paddle.arange(seq_len).unsqueeze(0).expand([batch_size, -1])
    prompt = paddle.randint(0, 1000, (batch_size, 64))  # 假设 prompt 长度为 64

    # 测试 Reference Model 的 log_probs 计算
    print("测试 Reference Model...")
    ref_log_probs_refs = reference_model_group.async_run_method_batch(
        method_name="forward",
        input_ids=[input_ids.numpy() for _ in range(batch_size)],
        attention_mask=[attention_mask.numpy() for _ in range(batch_size)],
        position_ids=[position_ids.numpy() for _ in range(batch_size)],
        prompt=[prompt.numpy() for _ in range(batch_size)],
    )
    ref_log_probs = ray.get(ref_log_probs_refs)
    print(f"Reference log_probs shape: {ref_log_probs[0].shape}")

    # 测试 Actor Model 的序列生成
    print("测试 Actor Model 生成...")
    generated_sequences_refs = actor_model_group.async_run_method_batch(
        method_name="generate_sequences",
        input_ids=[input_ids.numpy() for _ in range(batch_size)],
        attention_mask=[attention_mask.numpy() for _ in range(batch_size)],
        position_ids=[position_ids.numpy() for _ in range(batch_size)],
    )
    generated_sequences = ray.get(generated_sequences_refs)
    print(f"Generated sequences shape: {generated_sequences[0].shape}")

    # 测试 Actor Model 的 log_probs 计算
    print("测试 Actor Model log_probs...")
    actor_log_probs_refs = actor_model_group.async_run_method_batch(
        method_name="forward",
        input_ids=[input_ids.numpy() for _ in range(batch_size)],
        attention_mask=[attention_mask.numpy() for _ in range(batch_size)],
        position_ids=[position_ids.numpy() for _ in range(batch_size)],
        prompt=[prompt.numpy() for _ in range(batch_size)],
    )
    actor_log_probs = ray.get(actor_log_probs_refs)
    print(f"Actor log_probs shape: {actor_log_probs[0].shape}")

    # 测试 Reward Model
    print("测试 Reward Model...")
    rewards_refs = reward_model_group.async_run_method_batch(
        method_name="forward",
        input_ids=[input_ids.numpy() for _ in range(batch_size)],
        attention_mask=[attention_mask.numpy() for _ in range(batch_size)],
        position_ids=[position_ids.numpy() for _ in range(batch_size)],
    )
    rewards = ray.get(rewards_refs)
    print(f"Rewards shape: {rewards[0].shape}")

    print("所有测试完成！")
    ray.shutdown()

if __name__ == "__main__":
    main()