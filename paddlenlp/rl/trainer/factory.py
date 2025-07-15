"""
Model factory functions for Ray-based distributed training.
"""
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import copy
import paddle
import paddle.distributed as dist
from paddle import nn
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import PipelineLayer
from paddle.io import DataLoader, Dataset, DistributedBatchSampler
from rich.console import Console
from rich.table import Table
from paddlenlp.transformers import PretrainedModel, PretrainedTokenizer
from paddlenlp.transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PretrainedConfig,
)
from paddlenlp.rl.trainer.actor_trainer import ActorReferenceTrainer
from paddlenlp.rl.trainer.reward_trainer import RewardTrainer
from paddlenlp.rl.trainer.critic_trainer import CriticTrainer
from paddlenlp.rl.utils.config_utils import (
    DataArgument,
    ModelArgument,
    TrainingArguments,
)
from paddlenlp.rl.models.score_model import AutoModelForScore
from paddlenlp.rl.utils.reshard_utils import ReshardController
from paddlenlp.rl.utils.timer_utils import timers_scope_runtimer
from paddlenlp.transformers.configuration_utils import LlmMetaConfig
from paddlenlp.trl import llm_utils
from paddlenlp.utils.log import logger

from paddlenlp.data import DataCollator
from paddlenlp.datasets.rlhf_datasets.protocol import DataProto, TensorDict
from paddlenlp.generation import GenerationConfig
from paddlenlp.trainer.trainer import (
    EvalLoopOutput,
    EvalPrediction,
    ProgressCallback,
    ShardingOption,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    TrainOutput,
    logger,
    speed_metrics,
)

from paddlenlp.rl.utils.infer_utils import infer_guard
from paddlenlp.rl.utils.offload_utils import reload_and_offload_scope, reload_tensor_to_gpu
from paddlenlp.rl.utils.reshard_utils import ReshardController
from paddlenlp.rl.utils.timer_utils import TimerScope, TimerScopeManualLabel
from paddlenlp.rl.trainer.trainer_utils import (
    MuteDefaultFlowCallback,
    batch_retokenize,
    guard_set_args,
    is_same_tokenizer,
    process_row,
)

def create_actor_models(
    model_args: ModelArgument,
    data_args: DataArgument,
    training_args: TrainingArguments,
    common_config: Dict,
    reshard_controller: ReshardController = None,
):
    with timers_scope_runtimer("Actor model loading time"):
        # actor model
        actor_model_config: PretrainedConfig = AutoConfig.from_pretrained(
            model_args.actor_model_name_or_path,
            tensor_parallel_output=training_args.tensor_parallel_output,
            tensor_parallel_degree=training_args.tensor_parallel_degree,
            tensor_parallel_rank=training_args.tensor_parallel_rank,
            recompute_granularity=training_args.recompute_granularity,
            dtype=training_args.model_dtype,
            recompute=training_args.recompute,
            recompute_use_reentrant=training_args.recompute_use_reentrant,
            **common_config,
        )
        LlmMetaConfig.set_llm_config(actor_model_config, training_args)

        actor_model_config.use_fused_head_and_loss_fn = training_args.use_fused_head_and_loss_fn
        actor_model_config.set_attn_func = True
        actor_model_config.max_position_embeddings = data_args.max_length
        actor_model_config.use_sparse_head_and_loss_fn = False
        actor_model_config.seq_length = data_args.max_length
        actor_model_config.max_sequence_length = data_args.max_length
        logger.info(f"Loading Actor model with config:\n\t{actor_model_config}\n")

        if not training_args.autotuner_benchmark:
            actor_model = AutoModelForCausalLM.from_pretrained(
                model_args.actor_model_name_or_path, config=actor_model_config
            )
        else:
            actor_model = AutoModelForCausalLM.from_config(actor_model_config)
    
    actor_tokenizer = AutoTokenizer.from_pretrained(
        model_args.actor_model_name_or_path,
        model_max_length=data_args.max_length,
        padding_side="left",
        tokenizer_alpha=model_args.actor_tokenizer_alpha,
        use_fast=True,
    )
    if actor_tokenizer.pad_token_id is None:
        actor_tokenizer.pad_token_id = actor_tokenizer.eos_token_id
    llm_utils.init_chat_template(actor_tokenizer, model_args.actor_model_name_or_path, model_args.chat_template)

    return actor_model, actor_tokenizer


def create_actor_trainer(
    self,
    model: Union[PretrainedModel, nn.Layer] = None,
    model_eval: Union[PretrainedModel, nn.Layer] = None,
    criterion: nn.Layer = None,
    args: TrainingArguments = None,
    data_collator: Optional[DataCollator] = None,  # type: ignore
    train_dataset: Optional[Dataset] = None,
    eval_dataset: Union[Dataset, Dict[str, Dataset]] = None,
    tokenizer: Optional[PretrainedTokenizer] = None,
    compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
    callbacks: Optional[List[TrainerCallback]] = None,
    optimizers: Tuple[paddle.optimizer.Optimizer, paddle.optimizer.lr.LRScheduler] = (None, None),
    preprocess_logits_for_metrics: Optional[Callable[[paddle.Tensor, paddle.Tensor], paddle.Tensor]] = None,
    reshard_controller: Optional[ReshardController] = None,
):
    policy_training_args = copy.deepcopy(args)
    lr_scheduler = self.get_scheduler(policy_training_args)
    actor_trainer = ActorReferenceTrainer(
        model,
        criterion,
        policy_training_args,
        data_collator,
        train_dataset,
        eval_dataset,
        tokenizer,
        compute_metrics,
        callbacks,
        [None, lr_scheduler],
        preprocess_logits_for_metrics,
        reshard_controller,
    )
    actor_trainer.set_eval_model(model_eval)
    actor_trainer.timers = self.timers

    actor_trainer.add_callback(MuteDefaultFlowCallback)
    if not args.disable_tqdm:
        actor_trainer.pop_callback(ProgressCallback)
    return actor_trainer


def create_reference_models(
    model_args: ModelArgument,
    data_args: DataArgument,
    training_args: TrainingArguments,
    common_config: Dict,
    reshard_controller: ReshardController = None,
):
    """
    创建 Reference 模型（与 Actor 模型结构相同）
    """
    return create_actor_models(model_args, data_args, training_args, common_config, reshard_controller)

def create_reference_trainer(
    self,
    model: Union[PretrainedModel, nn.Layer] = None,
    criterion: nn.Layer = None,
    args: TrainingArguments = None,
    data_collator: Optional[DataCollator] = None,  # type: ignore
    train_dataset: Optional[Dataset] = None,
    eval_dataset: Union[Dataset, Dict[str, Dataset]] = None,
    tokenizer: Optional[PretrainedTokenizer] = None,
    compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
    callbacks: Optional[List[TrainerCallback]] = None,
    optimizers: Tuple[paddle.optimizer.Optimizer, paddle.optimizer.lr.LRScheduler] = (None, None),
    preprocess_logits_for_metrics: Optional[Callable[[paddle.Tensor, paddle.Tensor], paddle.Tensor]] = None,
):
    with guard_set_args(
        args,
        {
            "recompute": False,
            # "fp16_opt_level": "O1",
            "pipeline_parallel_degree": (
                args.pipeline_parallel_degree if isinstance(model, PipelineLayer) else 1
            ),  # workaround for pipeline parallel model check
        },
    ):
        reference_trainer = ActorReferenceTrainer(
            model,
            criterion,
            copy.deepcopy(args),
            data_collator,
            train_dataset,
            eval_dataset,
            tokenizer,
            compute_metrics,
            callbacks,
            optimizers,
            preprocess_logits_for_metrics,
        )
        if args.pipeline_parallel_degree > 1 or ShardingOption.FULL_SHARD in args.sharding:
            reference_trainer.init_train_model_opt(100, None, clear_master_weight=True)  # dummy max_steps

    reference_trainer.timers = self.timers

    return reference_trainer


def create_reward_models(
    model_args: ModelArgument,
    data_args: DataArgument,
    training_args: TrainingArguments,
    common_config: Dict,
):
    with timers_scope_runtimer("Reward model loading time"):
        reward_model_config = AutoConfig.from_pretrained(
            model_args.reward_model_name_or_path,
            tensor_parallel_output=False,
            tensor_parallel_degree=training_args.tensor_parallel_degree,
            tensor_parallel_rank=training_args.tensor_parallel_rank,
            dtype=training_args.model_dtype,
            recompute=training_args.critic_recompute,
            recompute_granularity=model_args.critic_recompute_granularity,
            recompute_use_reentrant=training_args.recompute_use_reentrant,
            **common_config,
        )
        LlmMetaConfig.set_llm_config(reward_model_config, training_args)
        reward_model_config.max_position_embeddings = data_args.max_length
        reward_model_config.use_sparse_head_and_loss_fn = False
        logger.info(f"Loading Reward model with config:\n\t{reward_model_config}\n")

        config = copy.deepcopy(reward_model_config)
        if training_args.eval_mode is not None:
            if training_args.eval_mode == "single":
                config.tensor_parallel_degree = -1
                config.tensor_parallel_rank = 0

        if not training_args.autotuner_benchmark:
            reward_model = AutoModelForScore.from_pretrained(
                model_args.reward_model_name_or_path,
                config=config,
                score_type="reward",
                do_normalize=False,
            )
        else:
            reward_model = AutoModelForScore.from_config(
                config,
                score_type="reward",
                do_normalize=False,
            )

    reward_tokenizer = AutoTokenizer.from_pretrained(
        model_args.reward_model_name_or_path,
        model_max_length=data_args.max_length,
        padding_side="right",
        tokenizer_alpha=model_args.reward_tokenizer_alpha,
        use_fast=True,
    )
    if reward_tokenizer.pad_token_id is None:
        reward_tokenizer.pad_token_id = reward_tokenizer.eos_token_id
    llm_utils.init_chat_template(reward_tokenizer, model_args.reward_model_name_or_path, model_args.chat_template)
    return reward_model, reward_tokenizer

def create_reward_trainer(
    self,
    model: Union[PretrainedModel, nn.Layer, str] = None,
    criterion: nn.Layer = None,
    args: TrainingArguments = None,
    data_collator: Optional[DataCollator] = None,  # type: ignore
    train_dataset: Optional[Dataset] = None,
    eval_dataset: Union[Dataset, Dict[str, Dataset]] = None,
    tokenizer: Optional[PretrainedTokenizer] = None,
    compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
    callbacks: Optional[List[TrainerCallback]] = None,
    optimizers: Tuple[paddle.optimizer.Optimizer, paddle.optimizer.lr.LRScheduler] = (None, None),
    preprocess_logits_for_metrics: Optional[Callable[[paddle.Tensor, paddle.Tensor], paddle.Tensor]] = None,
):
    with guard_set_args(
        args,
        {
            "recompute": False,
            # "fp16_opt_level": "O1",
            "pipeline_parallel_degree": (
                args.pipeline_parallel_degree if isinstance(model, PipelineLayer) else 1
            ),  # workaround for pipeline parallel model check
        },
    ):
        reward_trainer = RewardTrainer(
            model,
            criterion,
            copy.deepcopy(args),
            data_collator,
            train_dataset,
            eval_dataset,
            tokenizer,
            compute_metrics,
            callbacks,
            optimizers,
            preprocess_logits_for_metrics,
            reward_server=model,
        )

        if not self.args.use_rm_server:
            if args.pipeline_parallel_degree > 1 or ShardingOption.FULL_SHARD in args.sharding:
                reward_trainer.init_train_model_opt(100, None, clear_master_weight=True)  # dummy max_steps

    reward_trainer.timers = self.timers

    return reward_trainer

def create_critic_models(
    model_args: ModelArgument,
    data_args: DataArgument,
    training_args: TrainingArguments,
    common_config: Dict,
    reward_model,
):
    with timers_scope_runtimer("Critic model loading time"):
        reward_model_config = reward_model.config
        if model_args.critic_model_name_or_path is None:
            model_args.critic_model_name_or_path = model_args.reward_model_name_or_path
            critic_model = AutoModelForScore.from_config(
                reward_model_config,
                dtype=training_args.model_dtype,
                score_type="critic",
                do_normalize=False,
                clip_range_value=training_args.clip_range_value,
                **common_config,
            )
            if not training_args.autotuner_benchmark:
                critic_model.set_state_dict(reward_model.state_dict())
        else:
            if not training_args.autotuner_benchmark:
                critic_model = AutoModelForScore.from_pretrained(
                    model_args.critic_model_name_or_path,
                    config=reward_model_config,
                    score_type="critic",
                    do_normalize=False,
                    clip_range_value=training_args.clip_range_value,
                    **common_config,
                )
            else:
                critic_model = AutoModelForScore.from_config(
                    reward_model_config,
                    score_type="critic",
                    do_normalize=False,
                    clip_range_value=training_args.clip_range_value,
                    **common_config,
                )

    critic_tokenizer = AutoTokenizer.from_pretrained(
        model_args.critic_model_name_or_path,
        model_max_length=data_args.max_length,
        padding_side="left",
        tokenizer_alpha=model_args.reward_critic_tokenizer_alpha,
        use_fast=True,
    )
    if critic_tokenizer.pad_token_id is None:
        critic_tokenizer.pad_token_id = critic_tokenizer.eos_token_id
    llm_utils.init_chat_template(critic_tokenizer, model_args.critic_model_name_or_path, model_args.chat_template)

    if training_args.eval_mode is not None:
        config = copy.deepcopy(critic_model.config)
        if training_args.eval_mode == "single":
            config.tensor_parallel_degree = -1
            config.tensor_parallel_rank = 0
        with timers_scope_runtimer("Reward critic eval model loading time"):
            critic_eval_model = AutoModelForScore.from_config(config)
    else:
        critic_eval_model = None

    return critic_model, critic_eval_model, critic_tokenizer

def create_critic_trainer(
    self,
    model: Union[PretrainedModel, nn.Layer] = None,
    model_eval: Union[PretrainedModel, nn.Layer] = None,
    criterion: nn.Layer = None,
    args: TrainingArguments = None,
    data_collator: Optional[DataCollator] = None,  # type: ignore
    train_dataset: Optional[Dataset] = None,
    eval_dataset: Union[Dataset, Dict[str, Dataset]] = None,
    tokenizer: Optional[PretrainedTokenizer] = None,
    compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
    callbacks: Optional[List[TrainerCallback]] = None,
    optimizers: Tuple[paddle.optimizer.Optimizer, paddle.optimizer.lr.LRScheduler] = (None, None),
    preprocess_logits_for_metrics: Optional[Callable[[paddle.Tensor, paddle.Tensor], paddle.Tensor]] = None,
):
    value_training_args = copy.deepcopy(args)
    for attr_name in [
        "critic_learning_rate",
        "critic_weight_decay",
        "critic_lr_scheduler_type",
        "critic_warmup_ratio",
        "critic_recompute",
    ]:
        if getattr(value_training_args, attr_name, None) is not None:
            setattr(
                value_training_args,
                attr_name[len("critic_") :],
                getattr(value_training_args, attr_name),
            )
    lr_scheduler = self.get_scheduler(value_training_args)
    critic_trainer = CriticTrainer(
        model,
        criterion,
        value_training_args,
        data_collator,
        train_dataset,
        eval_dataset,
        tokenizer,
        compute_metrics,
        callbacks,
        [None, lr_scheduler],
        preprocess_logits_for_metrics,
    )

    critic_trainer.set_eval_model(model_eval)
    critic_trainer.timers = self.timers

    critic_trainer.add_callback(MuteDefaultFlowCallback)
    if not args.disable_tqdm:
        critic_trainer.pop_callback(ProgressCallback)
    return critic_trainer
