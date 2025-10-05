from typing import Any

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


class CLIPJepaModel:
    def __init__(
        self,
        config: config.JepaConfig,
        device: torch.device,
    ):
        self.config = config
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path=config.model_name,
            dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
            attn_implementation="flash_attention_2" if device.type == "cuda" else "sdpa",
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(
            config.model_name,
            max_pixels=config.max_pixels,
        )

        self.processor.tokenizer.add_special_tokens(
            {
                "additional_special_tokens": [
                    config.embed_start_token,
                    config.embed_end_token,
                ]
            }
        )
        self.model.resize_token_embeddings(len(self.processor.tokenizer))

        self.embed_start_token_id = self.processor.tokenizer.convert_tokens_to_ids(
            config.embed_start_token
        )
        self.embed_end_token_id = self.processor.tokenizer.convert_tokens_to_ids(
            config.embed_end_token
        )

    def embed_text(self, texts: list[str]):
        messages = [
            [{"role": "user", "content": [{"type": "text", "text": text}]}] for text in texts
        ]
        embedding = self._encode_text_with_prediction(messages)
        return embedding

    def embed_image(self, images: list[Image.Image]):
        messages = [
            [{"role": "user", "content": [{"type": "image", "image": image}]}] for image in images
        ]
        embedding = self._encode_text_with_prediction(messages)
        return embedding

    def _encode_text_with_prediction(self, messages: list[dict[str, Any]]) -> torch.Tensor:
        """
        Encode text and extract embedding at the last valid token position.
        Uses the LLM's self-attention to produce embeddings for the text span
        located between <EMBED> and </EMBED> tokens.
        The embedding is extracted from the hidden state at the </EMBED> token position.

        Args:
            messages: List of dictionaries representing the chat history/messages.

        Returns:
            Normalized Embedding at the last valid token position [B, H]
        """

        processed_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = qwen_vl_utils.process_vision_info(messages)
        inputs = self.processor(
            text=[
                self.config.embed_start_token + text + self.config.embed_end_token
                for text in processed_text
            ],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        out = self.model(**inputs, output_hidden_states=True)
        h_last = out.hidden_states[-1]  # [B, T+2, H]
        last_idx = inputs["input_ids"] == self.embed_end_token_id
        embedding = h_last[last_idx]  # [B, H]
        return F.normalize(embedding, dim=-1)


def get_lora_model(
    model: Qwen2_5_VLForConditionalGeneration, hyper_parameters: config.HyperParameters
) -> Qwen2_5_VLForConditionalGeneration:
    lora_config = peft.LoraConfig(
        r=hyper_parameters.lora_config.lora_rank,
        lora_alpha=hyper_parameters.lora_config.lora_alpha,
        lora_dropout=hyper_parameters.lora_config.lora_dropout,
        target_modules=hyper_parameters.lora_config.target_modules,
        use_dora=True,
        init_lora_weights="gaussian",
    )
    lora_model = peft.get_peft_model(model, lora_config)
    trainable_params, all_params = lora_model.get_nb_trainable_parameters()
    logger.info(
        f"Trainable portion: {trainable_params / all_params:.4f}, trainable params: {trainable_params}"
    )
    return lora_model
