import torch


# We use a large "Target" model and a small "Draft" model.
# They MUST share the same tokenizer for this to work easily.
# TARGET_MODEL_NAME = "facebook/opt-1.3b" 
TARGET_MODEL_NAME = "facebook/opt-2.7b" 
DRAFT_MODEL_NAME  = "facebook/opt-125m"

# TARGET_MODEL_NAME = "distilgpt2" 
# DRAFT_MODEL_NAME  = "gpt2-large"

'''
Supported models
Draft Models (small, fast):

    facebook/opt-125m (125M params) ⭐ Recommended
    gpt2 (124M params)
    distilgpt2 (82M params)

Target Models (large, accurate):
    facebook/opt-1.3b (1.3B params) ⭐ Recommended for CPU
    facebook/opt-2.7b (2.7B params)
    gpt2-medium (355M params)
    gpt2-large (774M params)

'''


# Automatically detect if we have a GPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DEVICE = "cpu"

# Generation Parameters
TEMPERATURE = 1.0
TOP_K = 50
TOP_P = 0.9