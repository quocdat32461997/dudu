from typing import List

from datasets import Dataset

# from lightning import Trainer
# from litgpt.lora import merge_lora_weights
from trl import GRPOConfig, GRPOTrainer

from engines.utils import DataFactory, RewardFunctionFactory, TrainerFactory

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


@TrainerFactory.register("grpo_trainer")
def grpo_trainer(
    reward_fns: List[str], train_dataset: Dataset, model="unsloth/gemma-3-4b-it"
):
    # Init trainer
    training_args = GRPOConfig(
        output_dir=None,
        learning_rate=1e-5,
        remove_unused_columns=False,  # to access the solution column in accuracy_reward
        gradient_accumulation_steps=16,
        num_train_epochs=1,
        bf16=True,
        # Parameters that control de data preprocessing
        max_completion_length=64,  # default: 256
        num_generations=4,  # default: 8
        # max_prompt_length=128,  # default: 512
        # Parameters related to reporting and saving
        report_to=["tensorboard"],
        logging_steps=10,
        # push_to_hub=True,
        save_strategy="steps",
        save_steps=10,
        use_cpu=True,
    )
    # training_args = GRPOConfig(output_dir=None, use_cpu=True)

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[
            RewardFunctionFactory.get(reward_fn) for reward_fn in reward_fns  # noqa
        ],  # noqa
        args=training_args,
        train_dataset=train_dataset,
    )

    trainer.train()
