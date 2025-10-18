import lightning as L
import torch
from torch import nn
from torchvision import transforms
from tqdm.auto import tqdm
import wandb

from src import config, data, model, metrics


class CLIPJepaTrainer(L.LightningModule):
    def __init__(
        self,
        hyper_parameters: config.HyperParameters,
        model_components: config.ModelComponents,
        loss_fn: nn.Module,
    ):
        super().__init__()
        self.hyper_parameters = hyper_parameters
        self.model_components = model_components
        self.llm_model = model_components.llm_model
        self.llm_projection = model_components.llm_projection
        self.vision_model = model_components.vision_model
        self.loss_fn = loss_fn

    def common_step(self, batch: data.Batch, prefix: str, batch_idx: int) -> torch.Tensor:
        text_hidden_output = model.embed_text(
            texts=batch.texts, model_components=self.model_components
        )
        text_embeddings = self.llm_projection(text_hidden_output)
        image_embeddings: torch.Tensor = self.vision_model(batch.images)
        loss: torch.Tensor = self.loss_fn(text_embeddings, image_embeddings)
        img_acc, cap_acc = metrics.top_k_accuracy(
            image_embedding=image_embeddings, text_embedding=text_embeddings, k=5
        )

        common_log_kwargs = {
            "on_step": True,
            "on_epoch": True,
            "batch_size": len(batch),
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

        if batch_idx == 0 and prefix == "valid":
            _log_images(
                batch,
                text_embeddings @ image_embeddings.T,
                self.model_components.inverse_transform,
                self.current_epoch,
            )

        return loss

    def training_step(self, batch: data.Batch, batch_idx: int) -> torch.Tensor:
        return self.common_step(batch, "train", batch_idx)

    def validation_step(self, batch: data.Batch, batch_idx: int):
        _ = self.common_step(batch, "valid", batch_idx)

    def configure_optimizers(self):
        llm_trainable_params = [p for p in self.llm_model.parameters() if p.requires_grad]
        llm_projection_trainable_params = list(self.llm_projection.parameters())
        vision_base_trainable_params = list(self.vision_model.base.parameters())
        vision_projection_trainable_params = list(self.vision_model.projection.parameters())

        loss_params = list(self.loss_fn.parameters())

        optimizer = torch.optim.AdamW(
            [
                {
                    "params": llm_trainable_params + llm_projection_trainable_params,
                    "lr": self.hyper_parameters.learning_rate,
                },
                {
                    "params": vision_base_trainable_params,
                    "lr": self.hyper_parameters.learning_rate / 10,
                },
                {
                    "params": vision_projection_trainable_params,
                    "lr": self.hyper_parameters.learning_rate,
                },
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
        if hasattr(self.llm_model, "save_pretrained"):
            self.llm_model.save_pretrained(self.hyper_parameters.output_dir)


def get_trainer(hyper_parameters: config.HyperParameters, device: torch.device):
    return L.Trainer(
        max_epochs=hyper_parameters.epochs if not hyper_parameters.debug else 1,
        logger=L.pytorch.loggers.WandbLogger(),
        log_every_n_steps=hyper_parameters.log_every_n_steps,
        gradient_clip_val=1.0,
        limit_train_batches=2 if hyper_parameters.debug else 1.0,
        limit_val_batches=100 if hyper_parameters.debug else 1.0,
        accelerator="auto",
        num_sanity_val_steps=0 if hyper_parameters.debug else 2,
        precision="bf16" if device.type in {"cuda", "mps"} else "32",
    )


def _log_images(
    batch: data.Batch,
    similarity_matrix: torch.Tensor,
    inverse_transform: transforms.Normalize,
    epoch: int,
):
    closest_image_indices = similarity_matrix.argmax(dim=-1)
    to_pil = transforms.ToPILImage()
    table = wandb.Table(
        columns=["text", "image", "similarity"],
        data=[
            [
                text,
                wandb.Image(to_pil(inverse_transform(batch.images[image_index]).cpu())),
                similarity_matrix[i, image_index].item(),
            ]
            for i, (text, image_index) in tqdm(enumerate(zip(batch.texts, closest_image_indices)))
        ],
    )
    wandb.log(
        {
            f"clip_jepa_images_{epoch}": table,
        }
    )
