import datetime

import lightning as L
import torch
from torch import nn
from PIL import Image

from src import config, model, metrics


class CLIPJepaTrainer(L.LightningModule):
    def __init__(
        self,
        hyper_parameters: config.HyperParameters,
        model_components: model.ModelComponents,
        loss_fn: nn.Module,
    ):
        super().__init__()
        self.hyper_parameters = hyper_parameters
        self.model_components = model_components
        self.loss_fn = loss_fn

    def common_step(self, batch: dict[str, list[str] | list[Image.Image]], prefix: str):
        text = batch["texts"]
        images = batch["images"]
        text_embeddings = model.embed_text(text, self.model_components)
        image_embeddings = model.embed_image(images, self.model_components)
        loss = self.loss_fn(text_embeddings, image_embeddings)
        img_acc, cap_acc = metrics.metrics(image_embeddings, text_embeddings)

        batch_size = len(text)
        common_log_kwargs = {
            "on_step": False,
            "on_epoch": True,
            "batch_size": batch_size,
            "prog_bar": True,
        }
        self.log(f"{prefix}_loss", loss, **common_log_kwargs)
        self.log(
            f"{prefix}_img_acc",
            img_acc,
            **common_log_kwargs,
        )
        self.log(
            f"{prefix}_cap_acc",
            cap_acc,
            **common_log_kwargs,
        )

        return loss

    def training_step(self, batch: dict[str, list[str] | list[Image.Image]], batch_idx: int):
        return self.common_step(batch, "train")

    def validation_step(self, batch: dict[str, list[str] | list[Image.Image]], batch_idx: int):
        _ = self.common_step(batch, "valid")

    def configure_optimizers(self):
        # Only LoRA (and any separate loss head) params
        trainable = [p for p in self.model_components.model.parameters() if p.requires_grad]
        # If you also want to train a custom loss head, include it here:
        loss_params = (
            list(self.loss_fn.parameters())
            if any(p.requires_grad for p in self.loss_fn.parameters())
            else []
        )

        optimizer = torch.optim.AdamW(
            [
                {"params": trainable, "lr": self.hyper_parameters.learning_rate},
                {"params": loss_params, "lr": self.hyper_parameters.learning_rate / 10},
            ]
        )

        if self.hyper_parameters.lr_scheduler:
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.hyper_parameters.learning_rate,
                total_steps=int(self.trainer.estimated_stepping_batches),
            )
            return [optimizer], [scheduler]
        else:
            return optimizer

    def on_save_checkpoint(self, checkpoint):
        # Save adapters into a folder, e.g., self.trainer.logger.log_dir or a hyperparam path
        # Works for both PeftModel and nonwrapped (no-op) if you guard it:
        if hasattr(self.model_components.model, "save_pretrained"):
            self.model_components.model.save_pretrained(self.hyper_parameters.output_dir)


def _get_wandb_logger(hyper_parameters: config.HyperParameters):
    name = f"{hyper_parameters.wandb_config.project}-{datetime.datetime.now()}"
    project = hyper_parameters.wandb_config.project
    if hyper_parameters.debug:
        name = "debug-" + name
        project = "debug-" + project

    return L.pytorch.loggers.WandbLogger(
        entity=hyper_parameters.wandb_config.entity,
        save_dir=hyper_parameters.wandb_config.wandb_log_path,
        project=project,
        name=name,
        config=hyper_parameters.model_dump(),
    )


def get_trainer(hyper_parameters: config.HyperParameters):
    return L.Trainer(
        max_epochs=hyper_parameters.epochs if not hyper_parameters.debug else 1,
        logger=_get_wandb_logger(hyper_parameters),
        log_every_n_steps=hyper_parameters.log_every_n_steps,
        gradient_clip_val=1.0,
        limit_train_batches=5 if hyper_parameters.debug else 1.0,
        limit_val_batches=5 if hyper_parameters.debug else 1.0,
        accelerator="auto",
    )
