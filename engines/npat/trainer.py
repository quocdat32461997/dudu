from typing import List

from engines.utils import TrainerFactory

# from lightning import Trainer
# from litgpt.lora import merge_lora_weights
from trl import GRPOConfig, GRPOTrainer
from utils import DataFactory, RewardFunctionFactory  # , ModelFactory

# @TrainerFactory.register("next_product_as_text")
# def train_fn():
#     # Init trainer, model, and data_module
#     data_module = DataFactory.get("next_product_as_text")
#     trainer = Trainer()

#     with trainer.init_module(empty_init=True):
#         model = ModelFactory.get("next_product_as_text")
#     trainer.fit(model, data_module)

#     # Save final checkpoint
#     merge_lora_weights(model.model)
#     trainer.save_checkpoint()


def create_grpo_trainer(
    experiment_name: str,
    reward_fns: List[str],
    train_dataset_name: str,
):
    @TrainerFactory.register(experiment_name)
    def trainer_fn():
        # Init trainer
        training_args = GRPOConfig(output_dir=None)

        trainer = GRPOTrainer(
            model="google/gemma-3-4b-it",
            reward_funcs=[
                RewardFunctionFactory.get(reward_fn) for reward_fn in reward_fns  # noqa
            ],  # noqa
            args=training_args,
            train_dataset=DataFactory.get(train_dataset_name),
        )

        trainer.train()

    return trainer_fn
