import sys
import os
import paddle
import ray

from functools import partial
from paddlenlp.datasets.rlhf_datasets.protocol import DataProto, TensorDict
from paddlenlp.datasets.rlhf_datasets import RLHFDataset, collate_fn
from paddlenlp.generation import GenerationConfig
from paddlenlp.utils.log import logger
from paddlenlp.trainer import PdArgumentParser, TrainingArguments
from paddlenlp.rl.trainer.ray.policy_actor import PolicyModelRayActor
from paddlenlp.rl.trainer.ray.critic_actor import CriticModelRayActor
from paddlenlp.rl.trainer.ray.reference_actor import ReferenceModelRayActor
from paddlenlp.rl.trainer.ray.reward_actor import RewardModelRayActor
from paddlenlp.rl.trainer.ray.launcher import RayActorGroup

from paddlenlp.rl.utils.config_utils import (
    DataArgument,
    ModelArgument,
    TrainingArguments,
)
from paddlenlp.rl.utils.offload_utils import offload_tensor_to_cpu
from paddlenlp.rl.utils.reshard_utils import ReshardController

def create_rl_dataset(data_args, training_args, tokenizer):
    requires_label = True if training_args.use_rm_server else False
    train_ds = RLHFDataset(
        dataset_name_or_path=data_args.train_datasets,
        tokenizer=tokenizer,
        max_prompt_len=data_args.max_prompt_len,
        requires_label=requires_label,
        prompt_key=data_args.prompt_key,
        response_key=data_args.response_key,
        splits="train",
    )
    dev_ds = RLHFDataset(
        dataset_name_or_path=data_args.eval_datasets,
        tokenizer=tokenizer,
        max_prompt_len=data_args.max_prompt_len,
        requires_label=requires_label,
        prompt_key=data_args.prompt_key,
        response_key=data_args.response_key,
        splits="dev",
    )
    return train_ds, dev_ds

def process_args(model_args: ModelArgument, data_args: DataArgument, training_args: TrainingArguments):
    training_args.max_src_len = data_args.max_prompt_len
    training_args.actor_model_name_or_path = model_args.actor_model_name_or_path
    training_args.max_length = data_args.max_length

    if training_args.use_rm_server:
        if model_args.reward_server is None:
            raise ValueError("Please specify reward_server when use_rm_server is true.")
        logger.info(f"Use reward server: {model_args.reward_server} for training.")
        if training_args.rl_algorithm == "ppo" and model_args.critic_model_name_or_path is None:
            raise ValueError("Please specify critic_model_name_or_path when use_rm_server is true.")
    else:
        if model_args.reward_model_name_or_path is None:
            raise ValueError("Please specify reward_model_name_or_path when use_rm_server is false.")

    training_args.print_config(model_args, "Model")
    training_args.print_config(data_args, "Data")

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, "
        f"world_size: {training_args.world_size}, " + f"distributed training: {bool(training_args.local_rank != -1)}, "
        f"16-bits training: {training_args.fp16 or training_args.bf16}"
    )
    return model_args, data_args, training_args

def main():
    # 参数解析
    parser = PdArgumentParser((ModelArgument, DataArgument, TrainingArguments))
    if len(sys.argv) >= 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file_and_cmd_lines()
    elif len(sys.argv) >= 2 and sys.argv[1].endswith(".yaml"):
        model_args, data_args, training_args = parser.parse_yaml_file_and_cmd_lines()
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 预处理参数
    model_args, data_args, training_args = process_args(model_args, data_args, training_args)

    # 构造common_config
    common_config = dict(
        use_flash_attention=getattr(training_args, "use_flash_attention", False),
        sequence_parallel=getattr(training_args, "sequence_parallel", False),
        fused_rotary=False,
        max_sequence_length=getattr(data_args, "max_length", 128),
    )

    # 构造reshard_controller
    if (
        getattr(training_args, "rollout_tensor_parallel_degree", 1) != getattr(training_args, "tensor_parallel_degree", 1)
        or getattr(training_args, "pipeline_parallel_degree", 1) > 1
    ):
        reshard_controller = ReshardController(tensor_parallel_degree=training_args.rollout_tensor_parallel_degree)
    else:
        reshard_controller = None

    args=training_args,
    # train_dataset=(train_ds if training_args.do_train and training_args.should_load_dataset else None),
    # eval_dataset=(dev_ds if training_args.do_eval and training_args.should_load_dataset else None),
    # actor_tokenizer=actor_tokenizer,
    # reference_tokenizer=actor_tokenizer,
    # reward_tokenizer=reward_tokenizer,
    # critic_tokenizer=critic_tokenizer,
    # data_collator=partial(
    #     collate_fn,
    #     pad_token_id=actor_tokenizer.pad_token_id,
    #     requires_label=True if training_args.use_rm_server else False,
    #     max_prompt_len=data_args.max_prompt_len if training_args.balance_batch else None,
    # ),  # NOTE: enforce prompt padding to max_prompt_len when using balance_batch
    def compute_metrics(eval_preds):
        accuracy = (eval_preds.predictions == 3).astype("float32").mean().item()
        return {"accuracy": accuracy}

    try:
        generation_config = GenerationConfig.from_pretrained(model_args.actor_model_name_or_path)
    except:
        logger.warning("Can't find generation config, so it will not use generation_config field in the model config")
        generation_config = None
    
    trainer_agrs = {
        # "model": None,
        "criterion": None,
        "args": training_args,
        "data_collator": None,
        "train_dataset": None,
        "eval_dataset": None,
        # "tokenizer": None,
        "compute_metrics": compute_metrics,
        "callbacks": None,
        "optimizers": (None, None),
        "preprocess_logits_for_metrics": None,
        }
    # === 以下为原有Ray Actor测试流程 ===
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
        strategy_config, model_args, data_args, training_args, common_config, reshard_controller, **trainer_agrs
    ))

    print("初始化 Actor Model...")
    ray.get(actor_model_group.async_init_model_from_pretrained(
        strategy_config, model_args, data_args, training_args, common_config, reshard_controller, **trainer_agrs
    ))

    print("初始化 Reward Model...")
    ray.get(reward_model_group.async_init_model_from_pretrained(
        strategy_config, model_args, data_args, training_args, common_config, reshard_controller, **trainer_agrs
    ))

    # 准备测试数据
    batch_size = 4
    seq_len = 128
    vocab_size = 1000 
    input_ids = paddle.randint(0, vocab_size, (batch_size, seq_len))
    label_ids = paddle.randint(0, vocab_size, (batch_size, seq_len))
    position_ids = paddle.arange(seq_len).unsqueeze(0).expand([batch_size, -1])
    prompt = paddle.randint(0, vocab_size, (batch_size, 64))  # 假设 prompt 长度为 64

    # 构造DataProto对象列表
    batch_list = []
    for i in range(batch_size):
        batch_dict = {
            "input_ids": input_ids[i:i+1],
            "label_ids": label_ids[i:i+1],
            "position_ids": position_ids[i:i+1],
            "prompt": prompt[i:i+1],
        }
        data_proto = DataProto.from_single_dict(batch_dict)
        batch_list.append(data_proto)

    print(f'{type(batch_list[0].batch)=}')
    # 测试 Reference Model 的 log_probs 计算
    print("测试 Reference Model log_probs...")
    ref_log_probs_refs = reference_model_group.async_run_method_batch(
        method_name="compute_logprob",
        batch=batch_list,
        key=["ref_log_probs"]*batch_size      # key参数也要是等长list
    )
    ref_log_probs = ray.get(ref_log_probs_refs)
    print(ref_log_probs)
    # [[DataProto, DataProto], [DataProto, DataProto]] = [List[DataProto]（compute_logprob 签名中的返回值）, List[DataProto]]
    for rayactor_idx, dataproto_list in enumerate(ref_log_probs):
        for i, dataproto in enumerate(dataproto_list):
            print(f"ReferenceModelRayActor {rayactor_idx} Sample {i} log_probs shape: {dataproto.batch['ref_log_probs'].shape}")
    
    # 测试 Policy Model 的序列生成
    # print("测试 Policy Model 生成...")
    # generated_sequences_refs = actor_model_group.async_run_method_batch(
    #     method_name="generate_sequences",
    #     prompt_only_batch=batch_list,
    #     # do_eval=[True]*batch_size
    # )
    # generated_sequences = ray.get(generated_sequences_refs)
    # for rayactor_idx, dataproto_list in enumerate(generated_sequences):
    #     for i, dataproto in enumerate(dataproto_list):
    #         print(f"PolicyModelRayActor {rayactor_idx} Sample {i} Generated sequences shape: {dataproto.batch['input_ids'].shape}")

    # 测试 Policy Model 的 log_probs 计算
    print("测试 Policy Model log_probs...")
    actor_log_probs_refs = actor_model_group.async_run_method_batch(
        method_name="compute_logprob",
        batch=batch_list,
        key=["log_probs"]*batch_size      # key参数也要是等长list
    )
    actor_log_probs = ray.get(actor_log_probs_refs)
    print(actor_log_probs)
    # [[DataProto, DataProto], [DataProto, DataProto]] = [List[DataProto]（compute_logprob 签名中的返回值）, List[DataProto]]
    for rayactor_idx, dataproto_list in enumerate(actor_log_probs):
        for i, dataproto in enumerate(dataproto_list):
            print(f"PolicyModelRayActor {rayactor_idx} Sample {i} log_probs shape: {dataproto.batch['log_probs'].shape}")

    # 测试 Reward Model
    print("测试 Reward Model...")
    rewards_refs = reward_model_group.async_run_method_batch(
        method_name="compute_reward",
        batch=batch_list
    )
    rewards = ray.get(rewards_refs)
    rewards = paddle.to_tensor(rewards)
    print(f"Rewards shape: {rewards[0].shape}")

    print("所有测试完成！")
    ray.shutdown()

if __name__ == "__main__":
    main()