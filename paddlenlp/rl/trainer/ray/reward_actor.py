import ray
from paddlenlp.rl.trainer.factory import create_reward_models, create_reward_trainer
from paddlenlp.transformers import AutoTokenizer
from paddlenlp.rl.trainer.ray import BaseModelRayActor

@ray.remote(num_gpus=1)
class RewardModelRayActor(BaseModelRayActor):
    def init_model_from_pretrained(self, strategy_config: dict, model_args, data_args, training_args, common_config, reshard_controller=None, **trainer_args):
        self._setup_distributed(strategy_config)
        # 1. 创建模型和分词器
        if not training_args.use_rm_server and model_args.reward_model_name_or_path is not None:
            self.model, self.tokenizer = create_reward_models(model_args, data_args, training_args, common_config)
        else:
            actor_tokenizer = AutoTokenizer.from_pretrained(
                model_args.actor_model_name_or_path,
                model_max_length=data_args.max_length,
                padding_side="left",
                tokenizer_alpha=model_args.actor_tokenizer_alpha,
                use_fast=True,
            )
            if actor_tokenizer.pad_token_id is None:
                actor_tokenizer.pad_token_id = actor_tokenizer.eos_token_id
            self.model, self.tokenizer = model_args.reward_server, actor_tokenizer
        # 2. 创建 Trainer（只做推理/评估）
        self.trainer = create_reward_trainer(
            model=self.model,
            tokenizer=self.tokenizer,
            **trainer_args,
        )
    
    def compute_reward(self, batch, *args, **kwargs):
        # 如果 Trainer 有 compute_reward 方法
        result = self.trainer.compute_reward(batch=batch, input_ids_tokenizer=self.tokenizer, *args, **kwargs)
        return result.numpy()
    
    def request_reward_server(self, *args, **kwargs):
        return self.trainer.request_reward_server(*args, **kwargs)