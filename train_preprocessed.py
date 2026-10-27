"""Train only from NPZ tensors produced by preprocess_dataset.py."""
from __future__ import annotations

import sys

import train


if __name__ == "__main__":
    if "--preprocessed" not in sys.argv:
        sys.argv.insert(1, "--preprocessed")
    train.main()
