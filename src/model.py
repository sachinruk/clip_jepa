from typing import Any
from dataclasses import dataclass

from loguru import logger
import peft
from PIL import Image
import qwen_vl_utils
import torch
import torch.nn.functional as F
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
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
) -> ModelComponents:
    """
    Initialize the CLIP-JEPA model and processor.

    Args:
        jepa_config: Configuration for the JEPA model
        device: Device to load the model on

    Returns:
        ModelComponents containing model, processor, and token IDs
    """
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        pretrained_model_name_or_path=jepa_config.model_name,
        dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        attn_implementation="flash_attention_2" if device.type == "cuda" else "sdpa",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(
        jepa_config.model_name,
        max_pixels=jepa_config.max_pixels,
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
    return F.normalize(embedding, dim=-1)


def get_lora_model(
    model: Qwen2_5_VLForConditionalGeneration,
    hyper_parameters: config.HyperParameters,
) -> Qwen2_5_VLForConditionalGeneration:
    lora_config = peft.LoraConfig(
        r=hyper_parameters.lora_config.lora_rank,
        lora_alpha=hyper_parameters.lora_config.lora_alpha,
        lora_dropout=hyper_parameters.lora_config.lora_dropout,
        target_modules=hyper_parameters.lora_config.target_modules,
        use_dora=True,
        init_lora_weights="gaussian",
        modules_to_save=hyper_parameters.lora_config.modules_to_save,
    )
    lora_model = peft.get_peft_model(model, lora_config)
    trainable_params, all_params = lora_model.get_nb_trainable_parameters()
    logger.info(
        f"Trainable portion: {trainable_params / all_params:.4f}, trainable params: {trainable_params}"
    )
    return lora_model


class GradMaskHook:
    def __init__(
        self, embed_start_token_id: int, embed_end_token_id: int, embed_shape: tuple[int, int]
    ):
        self.embed_start_token_id = embed_start_token_id
        self.embed_end_token_id = embed_end_token_id
        self.mask = torch.zeros(embed_shape, dtype=torch.bool)
        self.mask[embed_start_token_id] = True
        self.mask[embed_end_token_id] = True

    def __call__(self, grad: torch.Tensor) -> torch.Tensor:
        return grad * self.mask.float()


def embedding_zero_grad(lora_model: Qwen2_5_VLForConditionalGeneration, grad_hook: GradMaskHook):
    embed = lora_model.get_input_embeddings()
    embed.weight.requires_grad_(True)
    embed.weight.register_hook(grad_hook)
