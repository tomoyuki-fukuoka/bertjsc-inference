# Inference-Only Usage Guide

This guide is for users who only need to perform inference (prediction) with pre-trained BERTJSC models, without PyTorch Lightning dependency.

## Why Inference-Only Classes?

The inference-only classes (`BertForMaskedLMInference`, `SoftMaskedBertInference`) provide significant benefits for production deployments:

- **~500MB smaller deployments**: Removes PyTorch Lightning and its dependencies
- **Faster cold starts**: Less code to load and initialize
- **Reduced memory footprint**: No training-related components
- **Same functionality**: 100% compatible with pre-trained checkpoints

These classes are ideal for:
- Production API servers
- Serverless deployments (AWS Lambda, Google Cloud Run, etc.)
- Docker containers
- Edge devices with limited resources

## Quick Start

### 1. Basic Inference (BERT for Masked LM)

```python
import torch
from transformers import BertJapaneseTokenizer
from bertjsc import predict_of_json, BertForMaskedLMInference

# Initialize tokenizer and model
tokenizer = BertJapaneseTokenizer.from_pretrained(
    "cl-tohoku/bert-base-japanese-whole-word-masking"
)
model = BertForMaskedLMInference(
    card="cl-tohoku/bert-base-japanese-whole-word-masking"
)

# Load pre-trained checkpoint
checkpoint = torch.load('path/to/lit-bert-for-maskedlm-230112.pth', map_location='cpu')
model.load_state_dict(checkpoint, strict=False)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Perform inference
result = predict_of_json(model, tokenizer, device, "日本語校正してみす。")
print(result)
```

**Output:**
```json
{
    0: {'token': '日本語', 'score': 0.999341},
    1: {'token': '校', 'score': 0.996382},
    2: {'token': '正', 'score': 0.997387},
    3: {'token': 'し', 'score': 0.999978},
    4: {'token': 'て', 'score': 0.999999},
    5: {'token': 'み', 'score': 0.999947},
    6: {'token': 'す', 'correct': 'ます', 'score': 0.972711},
    7: {'token': '。', 'score': 1.0}
}
```

### 2. Soft-Masked BERT Inference

```python
import torch
from transformers import BertJapaneseTokenizer
from bertjsc import predict_of_json, SoftMaskedBertInference

# Initialize tokenizer and model
tokenizer = BertJapaneseTokenizer.from_pretrained(
    "cl-tohoku/bert-base-japanese-whole-word-masking"
)
model = SoftMaskedBertInference(
    card="cl-tohoku/bert-base-japanese-whole-word-masking",
    mask_token_id=tokenizer.mask_token_id,
    vocab_size=tokenizer.vocab_size
)

# Load pre-trained checkpoint
checkpoint = torch.load('path/to/lit-soft-masked-bert-230314.pth', map_location='cpu')
model.load_state_dict(checkpoint, strict=False)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Perform inference
result = predict_of_json(model, tokenizer, device, "日本語校正してみす。")
print(result)
```

## Model Files

Download pre-trained models:

- **BERT for Masked LM**: [lit-bert-for-maskedlm-230112.pth](https://drive.google.com/file/d/1IVcwz70GWWpOfJNG-Jm4jKjA9ObcrHJY/view?usp=sharing)
- **Soft-Masked BERT**: [lit-soft-masked-bert-230314.pth](https://drive.google.com/file/d/1uZQWq4gNszhmpFijNGHY8DB9ppuzHS7t/view?usp=sharing)

## API Reference

### BertForMaskedLMInference

Inference-only BERT for Masked Language Model.

**Constructor:**
```python
BertForMaskedLMInference(card: str)
```

**Parameters:**
- `card` (str): Hugging Face model card name (e.g., 'cl-tohoku/bert-base-japanese-whole-word-masking')

**Methods:**
- `forward(input_ids, attention_mask, token_type_ids, labels=None)`: Forward pass for inference
  - Returns: `transformers.modeling_outputs.MaskedLMOutput`

### SoftMaskedBertInference

Inference-only Soft-Masked BERT model.

**Constructor:**
```python
SoftMaskedBertInference(card: str, mask_token_id: int, vocab_size: int)
```

**Parameters:**
- `card` (str): Hugging Face model card name
- `mask_token_id` (int): ID of the mask token from tokenizer
- `vocab_size` (int): Size of the vocabulary from tokenizer

**Methods:**
- `forward(input_ids, token_type_ids, attention_mask, output_ids=None, det_labels=None)`: Forward pass for inference
  - Returns: `transformers.modeling_outputs.MaskedLMOutput`

## Migration from Lightning Models

If you're currently using `LitBertForMaskedLM` or `LitSoftMaskedBert`, migration is straightforward:

### Before (with PyTorch Lightning):
```python
from bertjsc.lit_model import LitBertForMaskedLM

model = LitBertForMaskedLM("cl-tohoku/bert-base-japanese-whole-word-masking")
```

### After (inference-only):
```python
from bertjsc import BertForMaskedLMInference

model = BertForMaskedLMInference(card="cl-tohoku/bert-base-japanese-whole-word-masking")
```

**Key Points:**
- Same initialization parameters
- Same checkpoint files work with both versions
- Same `forward()` interface
- No code changes needed in inference logic

## Performance Optimization

The inference-only classes include several optimizations:

1. **Cached mask token embedding**: The mask token embedding is computed once during initialization and reused, avoiding redundant computations on each forward pass.

2. **No training components**: Loss functions and training-related code are removed, reducing memory usage.

3. **Simplified architecture**: Only inference-critical components are included.

## Troubleshooting

### "Unexpected key(s) in state_dict" Error

If you encounter errors about `position_ids` when loading checkpoints:

```python
# Add strict=False to load_state_dict
model.load_state_dict(checkpoint, strict=False)
```

Or downgrade transformers:
```bash
pip install transformers==4.29.2
```

### CUDA Out of Memory

For large texts or limited GPU memory:

```python
# Use CPU for inference
device = torch.device('cpu')
model.to(device)

# Or process in smaller batches
```

## Docker Example

Example Dockerfile for inference-only deployment:

```dockerfile
FROM python:3.12-slim

# Install dependencies
RUN pip install --no-cache-dir torch transformers

# Install bertjsc (inference-only)
RUN pip install --no-cache-dir git+https://github.com/tomoyuki-fukuoka/bertjsc.git@feature/remove-pytorch-lightning-dependency

# Copy model files
COPY models/ /app/models/

# Your application code
COPY app.py /app/
WORKDIR /app

CMD ["python", "app.py"]
```

