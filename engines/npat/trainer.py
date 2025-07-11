from engines.utils import TrainerFactory
from lightning import Trainer
from litgpt.lora import merge_lora_weights
from utils import DataFactory, ModelFactory


@TrainerFactory.register("next_product_as_text")
def train_fn():
    # Init trainer, model, and data_module
    data_module = DataFactory.get("next_product_as_text")
    trainer = Trainer()

    with trainer.init_module(empty_init=True):
        model = ModelFactory.get("next_product_as_text")
    trainer.fit(model, data_module)

    # Save final checkpoint
    merge_lora_weights(model.model)
    trainer.save_checkpoint()
