from typing import Any
from dataclasses import dataclass

from loguru import logger
import peft
from PIL import Image
import qwen_vl_utils
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)

from src import config


@dataclass
class ModelComponents:
    """Container for model components."""

    model: Qwen2_5_VLForConditionalGeneration
    processor: AutoProcessor
    jepa_config: config.JepaConfig
    embed_start_token_id: int
    embed_end_token_id: int


def init_model(
    jepa_config: config.JepaConfig,
    device: torch.device,
    use_qlora: bool = False,
) -> ModelComponents:
    """
    Initialize the CLIP-JEPA model and processor.

    Args:
        jepa_config: Configuration for the JEPA model
        device: Device to load the model on
        use_qlora: Whether to use QLoRA (4-bit quantization)

    Returns:
        ModelComponents containing model, processor, and token IDs
    """
    # Setup quantization config for QLoRA
    quantization_config = None
    if use_qlora:
        logger.info("Using QLoRA (4-bit quantization)")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        pretrained_model_name_or_path=jepa_config.model_name,
        quantization_config=quantization_config,
        dtype=torch.bfloat16 if device.type in {"cuda", "mps"} else torch.float32,
        attn_implementation="flash_attention_2" if device.type == "cuda" else "sdpa",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(
        jepa_config.model_name,
        max_pixels=jepa_config.max_pixels,
        max_length=jepa_config.max_length,
    )

    processor.tokenizer.add_special_tokens(
        {
            "additional_special_tokens": [
                jepa_config.embed_start_token,
                jepa_config.embed_end_token,
            ]
        }
    )
    model.resize_token_embeddings(len(processor.tokenizer))

    embed_start_token_id = processor.tokenizer.convert_tokens_to_ids(jepa_config.embed_start_token)
    embed_end_token_id = processor.tokenizer.convert_tokens_to_ids(jepa_config.embed_end_token)

    return ModelComponents(
        model=model,
        processor=processor,
        jepa_config=jepa_config,
        embed_start_token_id=embed_start_token_id,
        embed_end_token_id=embed_end_token_id,
    )


def embed_text(
    texts: list[str],
    model_components: ModelComponents,
) -> torch.Tensor:
    """
    Embed text using the CLIP-JEPA model.

    Args:
        texts: List of text strings to embed
        model_components: ModelComponents containing model, processor, and config

    Returns:
        Normalized embeddings [B, H]
    """
    messages = [[{"role": "user", "content": [{"type": "text", "text": text}]}] for text in texts]
    return _encode_with_prediction(messages, model_components)


def embed_image(
    images: list[Image.Image],
    model_components: ModelComponents,
) -> torch.Tensor:
    """
    Embed images using the CLIP-JEPA model.

    Args:
        images: List of PIL images to embed
        model_components: ModelComponents containing model, processor, and config

    Returns:
        Normalized embeddings [B, H]
    """
    messages = [
        [{"role": "user", "content": [{"type": "image", "image": image}]}] for image in images
    ]
    return _encode_with_prediction(messages, model_components)


def _encode_with_prediction(
    messages: list[dict[str, Any]],
    model_components: ModelComponents,
) -> torch.Tensor:
    """
    Encode text and extract embedding at the last valid token position.
    Uses the LLM's self-attention to produce embeddings for the text span
    located between <EMBED> and </EMBED> tokens.
    The embedding is extracted from the hidden state at the </EMBED> token position.

    Args:
        messages: List of dictionaries representing the chat history/messages.
        model_components: ModelComponents containing model, processor, and config

    Returns:
        Normalized Embedding at the last valid token position [B, H]
    """

    processed_text = model_components.processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = qwen_vl_utils.process_vision_info(messages)
    inputs = model_components.processor(
        text=[
            model_components.jepa_config.embed_start_token
            + text
            + model_components.jepa_config.embed_end_token
            for text in processed_text
        ],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model_components.model.device)

    out = model_components.model(**inputs, output_hidden_states=True)
    h_last = out.hidden_states[-1]  # [B, T+2, H]
    last_idx = inputs["input_ids"] == model_components.embed_end_token_id
    embedding = h_last[last_idx]  # [B, H]
    # logger.info(f"Last hidden of {'image' if image_inputs else 'text'} state shape: {h_last.shape}")
    # if len(embedding) != len(messages):
    #     logger.warning(f"Embedding length mismatch: {len(embedding)} != {len(messages)}")
    return F.normalize(embedding, dim=-1)


def get_lora_model(
    model: Qwen2_5_VLForConditionalGeneration,
    hyper_parameters: config.HyperParameters,
) -> Qwen2_5_VLForConditionalGeneration:
    """
    Apply LoRA or QLoRA to the model.

    Args:
        model: The base model to apply LoRA to
        hyper_parameters: Configuration for LoRA
        use_qlora: If True, configures for QLoRA (no DoRA, prepares for gradient checkpointing)

    Returns:
        Model with LoRA adapters applied
    """
    # Prepare model for k-bit training if using QLoRA
    if hyper_parameters.lora_config.use_qlora:
        model = peft.prepare_model_for_kbit_training(model)
        logger.info("Model prepared for QLoRA (4-bit training)")

    lora_config = peft.LoraConfig(
        r=hyper_parameters.lora_config.lora_rank,
        lora_alpha=hyper_parameters.lora_config.lora_alpha,
        lora_dropout=hyper_parameters.lora_config.lora_dropout,
        target_modules=hyper_parameters.lora_config.target_modules,
        use_dora=hyper_parameters.lora_config.use_dora,  # DoRA doesn't work with quantized models
        init_lora_weights="gaussian",
        modules_to_save=hyper_parameters.lora_config.modules_to_save,
    )
    lora_model = peft.get_peft_model(model, lora_config)
    trainable_params, all_params = lora_model.get_nb_trainable_parameters()
    logger.info(
        f"{'QLoRA' if hyper_parameters.lora_config.use_qlora else 'LoRA'} applied - Trainable portion: {trainable_params / all_params:.4f}, trainable params: {trainable_params}"
    )
    return lora_model


class GradMaskHook:
    def __init__(
        self,
        embed_start_token_id: int,
        embed_end_token_id: int,
        embed_shape: tuple[int, int],
        device: torch.device,
    ):
        self.embed_start_token_id = embed_start_token_id
        self.embed_end_token_id = embed_end_token_id
        self.mask = torch.zeros(embed_shape, dtype=torch.bool, device=device)
        self.mask[embed_start_token_id] = True
        self.mask[embed_end_token_id] = True
        self.device = device

    def __call__(self, grad: torch.Tensor) -> torch.Tensor:
        if self.device.type in {"cuda", "mps"}:
            return grad * self.mask.bfloat16()
        else:
            return grad * self.mask.float()


def embedding_zero_grad(lora_model: Qwen2_5_VLForConditionalGeneration, grad_hook: GradMaskHook):
    embed = lora_model.get_input_embeddings()
    embed.weight.requires_grad_(True)
    embed.weight.register_hook(grad_hook)


class DeltaOnEmbedding(nn.Module):
    """
    Adds a (2, H) delta to the *output* of the input embedding only where
    input_ids == start_id or end_id. Keeps the base embedding frozen & intact.
    """

    def __init__(
        self,
        start_id: int,
        end_id: int,
        hidden_size: int,
        init_std: float = 0.02,
        dtype=None,
        device=None,
    ):
        super().__init__()
        self.start_id = int(start_id)
        self.end_id = int(end_id)
        self.delta = nn.Parameter(torch.zeros(2, hidden_size, dtype=dtype, device=device))
        nn.init.normal_(self.delta, mean=0.0, std=init_std)
        logger.info(
            f"Adding {self.delta.numel()} trainable parameters from DeltaOnEmbedding module."
        )

    def hook(self, embed_module: nn.Embedding, inputs, output):
        """
        Registered as a forward hook on the input embedding module.
        inputs[0]: input_ids  (B, T) or (T,)
        output:    embeddings (B, T, H) or (T, H)
        """
        input_ids = inputs[0]
        emb = output

        # Broadcast (..,1) so we can add a (H,) row delta
        mask_start = (input_ids == self.start_id).unsqueeze(-1)
        mask_end = (input_ids == self.end_id).unsqueeze(-1)

        # Add deltas only to those positions
        # cast masks to emb dtype for safe mixed-precision
        emb = (
            emb + mask_start.to(emb.dtype) * self.delta[0] + mask_end.to(emb.dtype) * self.delta[1]
        )
        return emb  # returning a new output is supported by forward hooks
