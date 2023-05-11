from typing import List, Optional, Tuple

import hydra
from hydra.core.config_store import ConfigStore
from hydra.types import TargetConf
import pyrootutils
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import Logger
from pytorch_lightning.callbacks import Callback
import torchvision.transforms as T
import numpy as np
from PIL import Image, ImageDraw

import torch
import os

class LogPredictionsCallback(Callback):
    def __init__(self, logger) -> None:
        super().__init__()
        self.wandb_logger = logger

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        """Called when the validation batch ends."""

        # `outputs` comes from `LightningModule.validation_step`
        # which corresponds to our model predictions in this case
        
        # Let's log 20 sample image predictions from first batch
        if batch_idx == 0:
            n = 16
            x, y = batch
            outputs = outputs['preds']
            images = x[:n]
            
            IMG_MEAN = [0.485, 0.456, 0.406]
            IMG_STD = [0.229, 0.224, 0.225]

            def denormalize(x, mean=IMG_MEAN, std=IMG_STD) -> torch.Tensor:
                # 3, H, W, B
                ten = x.clone().permute(1, 2, 3, 0)
                for t, m, s in zip(ten, mean, std):
                    t.mul_(s).add_(m)
                # B, 3, H, W
                return torch.clamp(ten, 0, 1).permute(3, 0, 1, 2)

            images = denormalize(images)
            images = images.cpu()
            outputs = outputs.clone().cpu()
            # images = images.cpu([])
            images_to_save = []
            for lm, img in zip(outputs, images):
                img = img.permute(1, 2, 0).numpy()*255
                h, w, _ = img.shape
                lm = (lm + 0.5) * np.array([w, h]) # convert to image pixel coordinates
                img = Image.fromarray(img.astype(np.uint8))
                draw = ImageDraw.Draw(img)
                for i in range(lm.shape[0]):
                    draw.ellipse((lm[i, 0] - 2, lm[i, 1] - 2,
                        lm[i, 0] + 2, lm[i, 1] + 2), fill=(255, 255, 0))
                images_to_save.append(img)

                # for j, img in enumerate(images):
                #     img = T.ToPILImage()(img)
                #     draw = ImageDraw.Draw(img)
                #     for i in range(outputs.shape[0]):
                #         draw.ellipse((outputs[j, i, 0] - 2, outputs[j, i, 1] - 2,
                #             outputs[j, i, 0] + 2, outputs[j, i, 1] + 2), fill=(255, 255, 0))
                #     images[j] = img
                #captions = [f'Ground Truth: {y_i} - Prediction: {y_pred}' for y_i, y_pred in zip(y[:n], outputs[:n])]
                
            # Option 1: log images with `WandbLogger.log_image`
            self.wandb_logger.log_image(key='sample_images', images=images_to_save)

            # Option 2: log predictions as a Table
            # columns = ['image', 'ground truth', 'prediction']
            # data = [[wandb.Image(x_i), y_i, y_pred] for x_i, y_i, y_pred in list(zip(x[:n], y[:n], outputs[:n]))]
            # wandb_logger.log_table(key='sample_table', columns=columns, data=data)

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
config_path = os.path.join(os.environ["PROJECT_ROOT"], "configs")
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/pyrootutils
# ------------------------------------------------------------------------------------ #

from src import utils

log = utils.get_pylogger(__name__)


@utils.task_wrapper
def train(cfg: DictConfig) -> Tuple[dict, dict]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator which applies extra utilities
    before and after the call.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = utils.instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))
    callbacks.append(LogPredictionsCallback(logger[0]))
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    if cfg.get("train"):
        log.info("Starting training!")
        print(cfg.get("ckpt_path"))
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))
        torch.save(model.state_dict(), 'model_state_dict.ckpt')

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path=config_path, config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:

    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = utils.get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )
    # return optimized metric
    return metric_value


if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    main()
