"""Utilities for data processing."""

from functools import partial
from typing import Any, Tuple

import torch
import torch.nn.functional as F
from tokenizers.processors import TemplateProcessing

from src.data.loading.components.interfaces import TokenizerConfig


def load_tokenize(config: TokenizerConfig) -> Any:
    """Load tokenizer and return a partial function for tokenization."""
    tokenizer = config.tokenizer
    if hasattr(config, "special_tokens"):
        tokenizer.add_special_tokens(config.special_tokens)
    if config.postprocess_eos_token:
        tokenizer._tokenizer.post_processor = TemplateProcessing(
            single="$A " + tokenizer.eos_token,
            special_tokens=[(tokenizer.eos_token, tokenizer.eos_token_id)],
        )
    tokenize = partial(
        tokenizer.encode_plus,
        max_length=config.max_length,
        padding=config.padding,
        truncation=config.truncation,
        add_special_tokens=config.add_special_tokens,
        return_tensors="pt",
    )
    return tokenize


def sample_gumbel(shape: Tuple, device: torch.device, eps=1e-20) -> torch.Tensor:
    """Sample from Gumbel(0, 1)"""
    U = torch.rand(shape, device=device)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + sample_gumbel(logits.shape, logits.device)
    sample = F.softmax(y / temperature, dim=-1)
    return sample
