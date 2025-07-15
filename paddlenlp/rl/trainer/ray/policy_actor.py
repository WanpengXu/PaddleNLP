import ray
from paddlenlp.rl.trainer.factory import create_actor_models, create_actor_trainer
from paddlenlp.transformers import AutoTokenizer
from paddlenlp.rl.trainer.ray import BaseModelRayActor

@ray.remote(num_gpus=1)
class PolicyModelRayActor(BaseModelRayActor):
    def init_model_from_pretrained(self, strategy_config: dict, pretrain, model_eval=None, reshard_controller=None, **trainer_args):
        self._setup_distributed(strategy_config)
        # 1. 创建模型和分词器
        self.model = create_actor_models(pretrain, strategy_config)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrain)
        # 2. 创建 Trainer
        self.trainer = create_actor_trainer(
            model=self.model,
            model_eval=model_eval,
            tokenizer=self.tokenizer,
            reshard_controller=reshard_controller,
            **trainer_args,  # 这里可以传入如 criterion, args, data_collator, train_dataset, eval_dataset 等
        )

    def fit(self, *args, **kwargs):
        # 训练入口，直接调用 Trainer 的训练方法
        return self.trainer.fit(*args, **kwargs)

    def save_model(self, *args, **kwargs):
        return self.trainer.save_model(*args, **kwargs)

    def generate_sequences(self, *args, **kwargs):
        # 生成序列，直接调用 Trainer 的 generate_sequences
        return self.trainer.generate_sequences(*args, **kwargs)

    def forward(self, *args, **kwargs):
        # 推理/采样等功能
        return self.trainer.forward(*args, **kwargs)