from typing import Any, List

import torch
from pytorch_lightning import LightningModule
from torchmetrics import MinMetric, MeanMetric
from torchmetrics.regression.mae import MeanAbsoluteError


class DlibLitModule(LightningModule):
    """Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 6 sections:
        - Computations (init)
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=['net'])
        # self.save_hyperparameters(logger=False)

        self.net = net

        # loss function
        self.criterion = torch.nn.MSELoss()

        # metric objects for calculating and averaging accuracy across batches
        self.train_mae = MeanAbsoluteError()
        self.val_mae = MeanAbsoluteError()
        self.test_mae = MeanAbsoluteError()

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_mae_best = MinMetric()

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.val_mae_best.reset()

    def model_step(self, batch: Any):
        x, y = batch
        preds = self.forward(x)
        loss = self.criterion(preds, y)
        # print("loss", loss, type(loss))
        # print("preds", preds, type(preds))
        # print("y", y, type(y))
        # exit(0)
        return loss, preds, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.train_mae(preds, targets)
        self.log("train/loss", self.train_loss,
                 on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/mae", self.train_mae, on_step=False,
                 on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`

        # Warning: when overriding `training_epoch_end()`, lightning accumulates outputs from all batches of the epoch
        # this may not be an issue when training on mnist
        # but on larger datasets/models it's easy to run into out-of-memory errors

        # consider detaching tensors before returning them from `training_step()`
        # or using `on_train_epoch_end()` instead which doesn't accumulate outputs

        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_mae(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False,
                 on_epoch=True, prog_bar=True)
        self.log("val/mae", self.val_mae, on_step=False,
                 on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        acc = self.val_mae.compute()  # get current val acc
        self.val_mae_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/mae_best", self.val_mae_best.compute(), prog_bar=True, sync_dist=True)

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.test_mae(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False,
                 on_epoch=True, prog_bar=True)
        self.log("test/mae", self.test_mae, on_step=False,
                 on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        _, preds, _ = self.model_step(batch)
        return preds

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    # read config file from configs/model/dlib_resnet.yaml
    import pyrootutils
    from omegaconf import DictConfig
    import hydra
    # from src.models.components.simple_resnet import SimpleResnet
    
    # find paths
    path = pyrootutils.find_root(
        search_from=__file__, indicator=".project-root")
    config_path = str(path / "configs" / "model")
    output_path = path / "outputs"
    pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
    print("paths", path, config_path, output_path)
    from src.models.components.simple_resnet import SimpleResnet

    def test_net(cfg):
        net = hydra.utils.instantiate(cfg.net)
        print("*"*20+" net "+"*"*20, "\n", net)
        output = net(torch.randn(16, 3, 224, 224))
        print("output", output.shape)

    def test_module(cfg):
        module = hydra.utils.instantiate(cfg)
        output = module(torch.randn(16, 3, 224, 224))
        print("module output", output.shape)

    @hydra.main(version_base="1.3", config_path=config_path, config_name="dlib_resnet.yaml")
    def main(cfg: DictConfig):
        # print(cfg)
        # test_net(cfg)
        # test_module(cfg)
        import albumentations as A
        from albumentations import Compose
        from albumentations.pytorch.transforms import ToTensorV2
        from matplotlib import pyplot as plt
        import numpy as np
        from PIL import Image, ImageDraw
        import cv2
        import pytorch_lightning as pl

        net = hydra.utils.instantiate(cfg.get('net'))
        # print(net)
        model = DlibLitModule.load_from_checkpoint(checkpoint_path='/Users/tiendzung/Downloads/checkpoints/epoch_095.ckpt', net = net)
        print(model)

        transform = Compose([
                A.Resize(224, 224),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        img = Image.open('/Users/tiendzung/Desktop/Screen Shot 2023-04-08 at 01.01.30.png').convert('RGB')
        w, h = img.size
        org_img = img.copy()
        landmarks = model.forward(transform(image = np.array(img))["image"].unsqueeze(0))[0]
        landmarks = (landmarks + 0.5) * torch.tensor([w, h])
        draw = ImageDraw.Draw(org_img)
        for i in range (landmarks.shape[0]):
            draw.ellipse((landmarks[i, 0] - 2, landmarks[i, 1] - 2,
                          landmarks[i, 0] + 2, landmarks[i, 1] + 2), fill=(255, 255, 0))
        plt.imshow(org_img)
        plt.show()

    main()
