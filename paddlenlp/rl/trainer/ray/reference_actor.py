import ray
from paddlenlp.rl.trainer.factory import create_reference_models, create_reference_trainer
from paddlenlp.transformers import AutoTokenizer
from paddlenlp.rl.trainer.ray import BaseModelRayActor

@ray.remote(num_gpus=1)
class ReferenceModelRayActor(BaseModelRayActor):
    def init_model_from_pretrained(self, strategy_config: dict, model_args, data_args, training_args, common_config, reshard_controller=None, **trainer_args):
        self._setup_distributed(strategy_config)
        # 1. 创建模型和分词器
        self.model, self.tokenizer = create_reference_models(model_args, data_args, training_args, common_config, reshard_controller)
        # 2. 创建 Trainer（只做推理/评估）
        self.trainer = create_reference_trainer(
            model=self.model,
            tokenizer=self.tokenizer,
            **trainer_args,
        )

    def compute_logprob(self, *args, **kwargs):
        return self.trainer.compute_logprob(*args, **kwargs)

    def generate_sequences(self, *args, **kwargs):
        return self.trainer.generate_sequences(*args, **kwargs)