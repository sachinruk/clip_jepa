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
    logger.info(f"Hyperparameters: {hyper_parameters.model_dump_json(indent=2)}")

    # Set environment variables
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Empty torch cache before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("Cleared CUDA cache")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        torch.mps.empty_cache()
        logger.info("Cleared MPS cache")

    # Create output directories
    hyper_parameters.output_dir.mkdir(parents=True, exist_ok=True)
    hyper_parameters.lora_config.lora_weight_path.mkdir(parents=True, exist_ok=True)
    hyper_parameters.wandb_config.wandb_log_path.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    logger.info("Loading datasets...")
    train_loader, valid_loader = data.get_dataset(hyper_parameters)
    logger.info(f"Train batches: {len(train_loader)}, Valid batches: {len(valid_loader)}")

    use_qlora = hyper_parameters.lora_config.use_qlora
    model_components = model.init_model(
        jepa_config=hyper_parameters.llm_model_config,
        device=device,
        use_qlora=use_qlora,
    )

    logger.info(f"Applying {'QLoRA' if use_qlora else 'LoRA'} adapters...")
    model_components.model = model.get_lora_model(
        model_components.model,
        hyper_parameters,
    )

    logger.info("Applying gradient mask to embedding weights...")
    embed_shape = model_components.model.get_input_embeddings().weight.shape
    grad_hook = model.GradMaskHook(
        embed_start_token_id=model_components.embed_start_token_id,
        embed_end_token_id=model_components.embed_end_token_id,
        embed_shape=embed_shape,
        device=device,
    )
    model.embedding_zero_grad(model_components.model, grad_hook)
    initial_embed_params = (
        model_components.model.get_input_embeddings().weight[:, :10].clone().detach()
    )
    initial_lm_head_params = model_components.model.lm_head.weight[:, :10].clone().detach()

    logger.info(f"Using loss type: {hyper_parameters.loss_type}")
    loss_fn = losses.get_loss(hyper_parameters.loss_type)

    logger.info("Creating Lightning trainer module...")
    lightning_module = trainer.CLIPJepaTrainer(
        hyper_parameters=hyper_parameters,
        model_components=model_components,
        loss_fn=loss_fn,
    )

    logger.info("Initializing Lightning trainer...")
    lightning_trainer = trainer.get_trainer(hyper_parameters, device)

    logger.info("Starting training...")
    lightning_trainer.fit(
        lightning_module,
        train_dataloaders=train_loader,
        val_dataloaders=valid_loader,
    )

    logger.success("Training completed!")
    logger.info(f"Model saved to: {hyper_parameters.output_dir}")

    final_embed_params = (
        model_components.model.get_input_embeddings().weight[:, :10].clone().detach()
    )
    final_lm_head_params = model_components.model.lm_head.weight[:, :10].clone().detach()
    logger.info(
        f"Initial embed params is equal to final embed params: {torch.allclose(initial_embed_params, final_embed_params)}"
    )
    logger.info(
        f"Initial lm head params is equal to final lm head params: {torch.allclose(initial_lm_head_params, final_lm_head_params)}"
    )
    logger.info(
        f"Initial embed params is equal to final lm head params: {torch.allclose(initial_embed_params, final_lm_head_params)}"
    )
    logger.info(
        f"Final embed params is equal to final lm head params: {torch.allclose(final_embed_params, final_lm_head_params)}"
    )
    logger.info("done")


if __name__ == "__main__":
    main()
