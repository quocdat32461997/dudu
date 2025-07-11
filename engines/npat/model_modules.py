import torch
from lightning import LightningModule
from litgpt.lora import GPT
from utils import ModelFactory


@ModelFactory.register("next_product_as_text")
class NextProductAsText(LightningModule):
    def __init__(self, *args, **kwargs):
        self.model = GPT.from_name(
            name="gemma-3-4b-it",
            lora_r=128,
            lora_alpha=256,
            lora_dropout=0.05,
            lora_query=True,
            lora_key=False,
            lora_value=True,
        )
        super().__init__(*args, **kwargs)

    def on_train_start(self):
        state_dict = torch.load("checkpoints/google/gemma-3-4b-it")
        self.model.load_state_dict(state_dict=state_dict, strict=False)

    def training_step(self, *args, **kwargs):
        return super().training_step(*args, **kwargs)

    def configure_optimizers(self):
        warmup_steps = 10
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=0.0002, weight_decay=0.0, betas=(0.9, 0.95)
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: step / warmup_steps
        )
        return [optimizer], [scheduler]
