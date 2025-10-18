from typing import Any

from loguru import logger
import peft
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers.utils.quantization_config import BitsAndBytesConfig
from torchvision import transforms

from src import config


class Projection(nn.Module):
    def __init__(self, d_in: int, d_out: int, p: float = 0.5) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_in, d_out, bias=False)
        self.linear2 = nn.Linear(d_out, d_out, bias=False)
        self.layer_norm = nn.LayerNorm(d_out)
        self.drop = nn.Dropout(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embed1 = self.linear1(x)
        embed2 = self.drop(self.linear2(F.silu(embed1)))
        embeds = self.layer_norm(embed1 + embed2)
        return embeds


class ProjectionLayers(nn.Module):
    def __init__(self, d_in: int, d_out: int, num_layers: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for _ in range(num_layers - 1):
            layers.extend([Projection(d_in, d_in), nn.SiLU()])
        layers += [Projection(d_in, d_out)]
        self.projection = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projection(x), dim=-1)


def init_model(
    jepa_config: config.JepaConfig,
    vision_model: nn.Module,
    inverse_transform: transforms.Normalize,
    device: torch.device,
    use_qlora: bool = False,
) -> config.ModelComponents:
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

    llm_model: transformers.modeling_utils.PreTrainedModel = (
        transformers.AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=jepa_config.model_name,
            quantization_config=quantization_config,
            dtype=torch.bfloat16 if device.type in {"cuda", "mps"} else torch.float32,
            attn_implementation="flash_attention_2" if device.type == "cuda" else "sdpa",
            device_map="auto",
        )
    )
    llm_projection = ProjectionLayers(
        llm_model.config.hidden_size, jepa_config.embed_dims, jepa_config.projection_layers
    )
    tokenizer: transformers.tokenization_utils.PreTrainedTokenizer = (
        transformers.AutoTokenizer.from_pretrained(
            jepa_config.model_name,
            max_pixels=jepa_config.max_pixels,
            max_length=jepa_config.max_length,
        )
    )

    tokenizer.add_special_tokens(
        {
            "additional_special_tokens": [
                jepa_config.embed_start_token,
                jepa_config.embed_end_token,
            ]
        }
    )
    llm_model.resize_token_embeddings(len(tokenizer))

    embed_start_token_id: int = tokenizer.convert_tokens_to_ids(jepa_config.embed_start_token)
    embed_end_token_id: int = tokenizer.convert_tokens_to_ids(jepa_config.embed_end_token)

    return config.ModelComponents(
        llm_model=llm_model,
        llm_projection=llm_projection,
        tokenizer=tokenizer,
        vision_model=vision_model,
        inverse_transform=inverse_transform,
        jepa_config=jepa_config,
        embed_start_token_id=embed_start_token_id,
        embed_end_token_id=embed_end_token_id,
    )


def embed_text(
    texts: list[str],
    model_components: config.ModelComponents,
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
    messages = [[{"role": "user", "content": [{"type": "text", "text": text}]}] for text in texts]
    processed_text: list[str] = model_components.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    text: list[str] = [
        model_components.jepa_config.embed_start_token
        + text
        + model_components.jepa_config.embed_end_token
        for text in processed_text
    ]
    inputs = model_components.tokenizer(
        text=text,
        padding=True,
        return_tensors="pt",
    ).to(model_components.llm_model.device)  # type: ignore

    out = model_components.llm_model(**inputs, output_hidden_states=True)
    h_last: torch.Tensor = out.hidden_states[-1]  # [B, T+2, H]
    last_idx = inputs["input_ids"] == model_components.embed_end_token_id
    return h_last[last_idx]  # [B, H]


def get_lora_model(
    model: transformers.modeling_utils.PreTrainedModel,
    hyper_parameters: config.HyperParameters,
) -> transformers.modeling_utils.PreTrainedModel:
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
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.start_id = start_id
        self.end_id = end_id
        self.delta = nn.Parameter(torch.zeros(2, hidden_size, dtype=dtype, device=device))
        nn.init.normal_(self.delta, mean=0.0, std=init_std)

    def hook(
        self, _: nn.Embedding, inputs: tuple[torch.Tensor, ...], output: torch.Tensor
    ) -> torch.Tensor:
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
