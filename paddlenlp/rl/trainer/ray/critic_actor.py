import ray
from paddlenlp.rl.trainer.factory import create_critic_models, create_critic_trainer
from paddlenlp.transformers import AutoTokenizer
from paddlenlp.rl.trainer.ray import BaseModelRayActor

@ray.remote(num_gpus=1)
class CriticModelRayActor(BaseModelRayActor):
    def init_model_from_pretrained(self, strategy_config: dict, model_args, data_args, training_args, common_config, reshard_controller=None, **trainer_args):
        self._setup_distributed(strategy_config)
        # 1. 创建模型和分词器
        self.model, self.tokenizer = create_critic_models(model_args, data_args, training_args, common_config, reshard_controller)
        # 2. 创建 Trainer
        self.trainer = create_critic_trainer(
            model=self.model,
            tokenizer=self.tokenizer,
            **trainer_args,  # 这里可以传入如 criterion, args, data_collator, train_dataset, eval_dataset 等
        )

    def fit(self, *args, **kwargs):
        # 训练入口，直接调用 Trainer 的训练方法
        return self.trainer.fit(*args, **kwargs)

    def save_model(self, *args, **kwargs):
        return self.trainer.save_model(*args, **kwargs)

    def forward(self, *args, **kwargs):
        # 推理/评估等功能
        return self.trainer.forward(*args, **kwargs)

    # 你可以根据需要添加更多方法，比如 evaluate、reload_states 等