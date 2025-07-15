import ray
from paddlenlp.rl.trainer.factory import create_reward_models, create_reward_trainer
from paddlenlp.transformers import AutoTokenizer
from paddlenlp.rl.trainer.ray import BaseModelRayActor

@ray.remote(num_gpus=1)
class RewardModelRayActor(BaseModelRayActor):
    def init_model_from_pretrained(self, strategy_config: dict, pretrain, model_eval=None, reshard_controller=None, **trainer_args):
        self._setup_distributed(strategy_config)
        # 1. 创建模型和分词器
        self.model = create_reward_models(pretrain, strategy_config)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrain)
        # 2. 创建 Trainer（只做推理/评估）
        self.trainer = create_reward_trainer(
            model=self.model,
            tokenizer=self.tokenizer,
            **trainer_args,
        )

    def forward(self, *args, **kwargs):
        # 只暴露推理相关接口
        return self.trainer.forward(*args, **kwargs)

    def compute_reward(self, *args, **kwargs):
        # 如果 Trainer 有 compute_reward 方法
        return self.trainer.compute_reward(*args, **kwargs)