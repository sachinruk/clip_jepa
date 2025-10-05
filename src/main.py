import json
import os

import click
import torch
from loguru import logger

from src import config, data, losses, model, trainer


@click.command()
@click.option(
    "--hyper-parameters-json",
    default="{}",
    help="JSON string containing hyperparameters to override defaults",
)
def main(hyper_parameters_json: str):
    """
    Main training function for CLIP-JEPA model.

    Pass hyperparameters as a JSON string to override defaults.
    Example:
        python -m src.main --hyper-parameters-json '{"epochs": 10, "batch_size": 16}'
    """
    # Parse hyperparameters
    logger.info("Parsing hyperparameters...")
    hyper_parameters = config.HyperParameters.model_validate_json(hyper_parameters_json)
    logger.info(f"Hyperparameters: {hyper_parameters.model_dump()}")

    # Set environment variables
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Create output directories
    hyper_parameters.output_dir.mkdir(parents=True, exist_ok=True)
    hyper_parameters.lora_config.lora_weight_path.mkdir(parents=True, exist_ok=True)
    hyper_parameters.wandb_config.wandb_log_path.mkdir(parents=True, exist_ok=True)

    # Detect device, supporting CUDA, MPS (Apple Silicon), or CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    # Load model
    logger.info("Loading CLIP-JEPA model...")
    clip_jepa_model = model.CLIPJepaModel(
        config=hyper_parameters.llm_model_config,
        device=device,
    )

    # Apply LoRA
    logger.info("Applying LoRA adapters...")
    lora_model = model.get_lora_model(clip_jepa_model.model, hyper_parameters)

    # Get loss function
    logger.info(f"Using loss type: {hyper_parameters.loss_type}")
    loss_fn = losses.get_loss(hyper_parameters.loss_type)

    # Get dataloaders
    logger.info("Loading datasets...")
    train_loader, valid_loader = data.get_dataset(hyper_parameters)
    logger.info(f"Train batches: {len(train_loader)}, Valid batches: {len(valid_loader)}")

    # Create Lightning module
    logger.info("Creating Lightning trainer module...")
    lightning_module = trainer.CLIPJepaTrainer(
        hyper_parameters=hyper_parameters,
        model=lora_model,
        loss_fn=loss_fn,
    )

    # Create Lightning trainer
    logger.info("Initializing Lightning trainer...")
    lightning_trainer = trainer.get_trainer(hyper_parameters)

    # Start training
    logger.info("Starting training...")
    lightning_trainer.fit(
        lightning_module,
        train_dataloaders=train_loader,
        val_dataloaders=valid_loader,
    )

    logger.success("Training completed!")
    logger.info(f"Model saved to: {hyper_parameters.output_dir}")


if __name__ == "__main__":
    main()
