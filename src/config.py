import pathlib

import pydantic


class JepaConfig(pydantic.BaseModel):
    model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    max_pixels: int = 384 * 384
    embed_start_token: str = "<EMBED>"
    embed_end_token: str = "</EMBED>"
    max_length: int = 1024

    class Config:
        extra = "forbid"


class WandbConfig(pydantic.BaseModel):
    project: str = "clip-jepa"
    entity: str = "sachinruk"
    wandb_log_path: pathlib.Path = pathlib.Path("/tmp/wandb")


class LoraConfig(pydantic.BaseModel):
    use_qlora: bool = True
    use_dora: bool = False
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: list[str] = [
        "qkv",
        "fc1",
        "fc2",
        "linear",
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    lora_weight_path: pathlib.Path = pathlib.Path("/tmp/lora")
    modules_to_save: None | list[str] = ["embed_tokens"]


class DataConfig(pydantic.BaseModel):
    dataset: str = "sayakpaul/coco-30-val-2014"
    num_workers: int = 4
    pin_memory: bool = True
    test_size: float = 0.1


class HyperParameters(pydantic.BaseModel):
    epochs: int = 5
    seed: int = 42
    batch_size: int = 8
    learning_rate: float = 5e-4
    lr_scheduler: bool = True
    accumulate_grad_batches: int = 1
    loss_type: str = "cyclip_sigmoid"
    temperature: float = -1.0
    log_every_n_steps: int = 50
    llm_model_config: JepaConfig = JepaConfig()
    wandb_config: WandbConfig = WandbConfig()
    lora_config: LoraConfig = LoraConfig()
    data_config: DataConfig = DataConfig()
    debug: bool = False
    output_dir: pathlib.Path = pathlib.Path("/tmp/output")

    class Config:
        extra = "forbid"
