# CLIP-JEPA

A PyTorch implementation of CLIP-style contrastive learning using a Joint Embedding Predictive Architecture (JEPA) with the Qwen2.5-VL multimodal model and LoRA fine-tuning.

## Overview

CLIP-JEPA combines the contrastive learning approach of CLIP with a vision-language model (Qwen2.5-VL) to learn aligned embeddings between images and text. The model uses LoRA (Low-Rank Adaptation) for efficient fine-tuning and supports multiple contrastive loss functions.

## Features

- **Multimodal Learning**: Learns joint embeddings for images and text using Qwen2.5-VL-3B-Instruct
- **Efficient Fine-tuning**: Uses LoRA (DoRA variant) for parameter-efficient training
- **Multiple Loss Functions**: Supports CLIP, CyCLIP, SigLIP, and CySigLIP losses
- **Hardware Flexibility**: Automatically detects and uses CUDA, Apple Silicon (MPS), or CPU
- **PyTorch Lightning Integration**: Streamlined training with distributed support
- **Weights & Biases Logging**: Built-in experiment tracking and monitoring

## Architecture

The model architecture consists of:

1. **Vision-Language Backbone**: Qwen2.5-VL-3B-Instruct (frozen or LoRA-adapted)
2. **Embedding Extraction**: Uses special `<EMBED>` and `</EMBED>` tokens to extract embeddings from the last hidden layer
3. **Normalization**: L2-normalized embeddings for contrastive learning
4. **Loss Functions**: Various (clip based) contrastive and cyclic loss formulations

### Embedding Process

- Text and images are processed through the Qwen2.5-VL model
- Special tokens (`<EMBED>`, `</EMBED>`) wrap the input
- Embeddings are extracted at the `</EMBED>` token position
- Embeddings are L2-normalized for cosine similarity calculations

## Installation

### Using the setup script (Recommended)

```bash
# Create virtual environment and install dependencies
./setup.sh

# Update existing environment
./setup.sh --update
```

### Manual installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Training
Use the debugger with `Train` since I have set this up via `launch.json`.

### Custom Hyperparameters

Pass hyperparameters as a JSON string:

```bash
python -m src.main --hyper-parameters-json '{
  "epochs": 10,
  "batch_size": 16,
  "learning_rate": 1e-4,
  "loss_type": "cyclip_sigmoid"
}'
```

### Configuration Options

All hyperparameters can be configured via JSON. Key options include:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `epochs` | 5 | Number of training epochs |
| `batch_size` | 8 | Training batch size |
| `learning_rate` | 5e-4 | Learning rate |
| `loss_type` | `"cyclip_sigmoid"` | Loss function (clip, cyclip, sigmoid, cyclip_sigmoid) |
| `temperature` | -1.0 | Learnable temperature parameter (sigmoid applied) |
| `accumulate_grad_batches` | 1 | Gradient accumulation steps |

See `src/config.py` for all available configuration options.

## Loss Functions

### CLIP Loss
Standard contrastive loss with symmetric image-to-text and text-to-image objectives.

### CyCLIP Loss
Adds cycle-consistency regularization:
- **Symmetry loss**: Encourages similarity matrix symmetry
- **Modality difference loss**: Aligns within-modality similarity matrices

### SigLIP Loss
Uses sigmoid-based binary cross-entropy instead of softmax for pairwise similarities.

### CySigLIP Loss
Combines sigmoid loss with cycle-consistency regularization.

## Model Configuration

### LoRA Settings

```python
lora_rank: 32
lora_alpha: 8
lora_dropout: 0.05
target_modules: ["qkv", "fc1", "fc2", "linear", "q_proj", "k_proj", "v_proj", 
                 "o_proj", "gate_proj", "up_proj", "down_proj"]
```

### Model Settings

```python
model_name: "Qwen/Qwen2.5-VL-3B-Instruct"
max_pixels: 262144  # 512 x 512
embed_start_token: "<EMBED>"
embed_end_token: "</EMBED>"
```

## Dataset

By default, uses the `sayakpaul/coco-30-val-2014` dataset. Configure in hyperparameters:

```json
{
  "data_config": {
    "dataset": "your-dataset-name",
    "test_size": 0.1,
    "num_workers": 4
  }
}
```

## Training Outputs

Training artifacts are saved to:
- **Model checkpoints**: `/tmp/output/` (configurable via `output_dir`)
- **LoRA weights**: `/tmp/lora/` (configurable via `lora_weight_path`)
- **Wandb logs**: `/tmp/wandb/` (configurable via `wandb_log_path`)

## Project Structure

```
clip_jepa/
├── src/
│   ├── config.py      # Configuration classes and hyperparameters
│   ├── data.py        # Dataset loading and dataloaders
│   ├── losses.py      # Loss function implementations
│   ├── main.py        # Training entry point
│   ├── metrics.py     # Evaluation metrics
│   ├── model.py       # CLIP-JEPA model implementation
│   └── trainer.py     # PyTorch Lightning trainer
├── notebooks/
│   ├── data_testing.ipynb
│   └── qwen_testing.ipynb
├── requirements.txt   # Python dependencies
├── setup.sh          # Setup script
└── README.md
```

## Requirements

- Python 3.8+
- PyTorch
- transformers
- peft (LoRA)
- lightning (PyTorch Lightning)
- datasets (Hugging Face)
- wandb
- qwen_vl_utils
- And more (see `requirements.txt`)

## Hardware Requirements

- **Minimum**: CPU (slow training)
- **Recommended**: NVIDIA GPU with CUDA support
- **Apple Silicon**: MPS backend supported for M1/M2/M3 Macs

The model automatically selects the best available device.

## Monitoring

Training metrics are logged to Weights & Biases:

```python
wandb_config:
  project: "clip-jepa"
  entity: "sachinruk"
```

Configure your W&B settings in the hyperparameters JSON.

## Development

### Testing Data Loading

```bash
python -m src.data
```

### Notebooks

Explore the notebooks for experimentation:
- `notebooks/data_testing.ipynb` - Dataset exploration
- `notebooks/qwen_testing.ipynb` - Model testing

## License

[Add your license here]

## Citation

If you use this code in your research, please cite:

```bibtex
[Add citation if applicable]
```

## Acknowledgments

- Built on [Qwen2.5-VL](https://huggingface.co/Qwen) from Alibaba Cloud
- Inspired by CLIP (OpenAI), CyCLIP, and SigLIP research
- Uses LoRA implementation from [PEFT](https://github.com/huggingface/peft)
