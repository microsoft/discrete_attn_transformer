# dat_common.py: common data structures and functions for tf_explorer2 program (and related modules)
import torch

EMBED_DATA_TYPE = torch.float16  # torch.uint8   #  
WEIGHT_TYPE = "one_hot"    # "one_hot" or "direct" (direct is experimental for now)
DEFAULT_D_REG = 96     # fast value that handles small to medium size examples   # MAX_VOCAB_LEN

