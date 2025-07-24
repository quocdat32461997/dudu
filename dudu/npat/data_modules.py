from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset
from utils import DataFactory


class AmazonDataset(Dataset):
    pass


@DataFactory.register("next_product_as_text")
class NextProductAsTextDataModule(LightningDataModule):
    def __init__(self):
        super().__init__()

    def setup(self, stage):
        return super().setup(stage)

    def train_dataloader(self):
        return super().train_dataloader()

    def val_dataloader(self):
        return super().val_dataloader()
