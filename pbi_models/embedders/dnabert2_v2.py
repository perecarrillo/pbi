from pbi_models.embedders.abstract_model import AbstractModel
import torch
from pbi_utils.embeddings_merging_strategies.abstract_merger_strategy import (
    AbstractMergerStrategy,
)
from pbi_utils.embeddings_merging_strategies.truncate_strategy import TruncateStrategy
from pbi_utils.logging import Logging, logging
from transformers import AutoTokenizer, AutoModel
from transformers.models.bert.configuration_bert import BertConfig

import glob
import os

logger = Logging()


def _patch_triton_syntax_if_needed():
    """Automatically patch bert_layers.py and flash_attn_triton.py in HF cache to use PyTorch native attention and valid Triton syntax."""
    hf_cache_dir = os.path.expanduser("~/.cache/huggingface/modules/transformers_modules")
    
    # Patch bert_layers.py to use native PyTorch attention
    bert_layers_pattern = os.path.join(hf_cache_dir, "**", "bert_layers.py")
    for filepath in glob.glob(bert_layers_pattern, recursive=True):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            modified_bert = False
            if "from .flash_attn_triton import flash_attn_qkvpacked_func" in content:
                content = content.replace(
                    "from .flash_attn_triton import flash_attn_qkvpacked_func",
                    "flash_attn_qkvpacked_func = None # Disabled legacy Triton import"
                )
                modified_bert = True
            if "attention_scores = torch.matmul(q, k)" in content:
                old_matmul_block = (
                    "            q = qkv[:, :, 0, :, :].permute(0, 2, 1, 3)  # b h s d\n"
                    "            k = qkv[:, :, 1, :, :].permute(0, 2, 3, 1)  # b h d s\n"
                    "            v = qkv[:, :, 2, :, :].permute(0, 2, 1, 3)  # b h s d\n"
                    "            attention_scores = torch.matmul(q, k) / math.sqrt(\n"
                    "                self.attention_head_size)\n"
                    "            attention_scores = attention_scores + bias\n"
                    "            attention_probs = nn.functional.softmax(attention_scores, dim=-1)\n"
                    "            attention_probs = self.dropout(attention_probs)\n"
                    "            attention = torch.matmul(attention_probs, v).permute(0, 2, 1,\n"
                    "                                                                 3)  # b s h d"
                )
                new_sdpa_block = (
                    "            q = qkv[:, :, 0, :, :].permute(0, 2, 1, 3)  # b h s d\n"
                    "            k = qkv[:, :, 1, :, :].permute(0, 2, 1, 3)  # b h s d\n"
                    "            v = qkv[:, :, 2, :, :].permute(0, 2, 1, 3)  # b h s d\n"
                    "            attention = nn.functional.scaled_dot_product_attention(\n"
                    "                q, k, v, attn_mask=bias, dropout_p=self.p_dropout if self.training else 0.0\n"
                    "            ).permute(0, 2, 1, 3)  # b s h d"
                )
                if old_matmul_block in content:
                    content = content.replace(old_matmul_block, new_sdpa_block)
                    modified_bert = True

            if modified_bert:
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(content)
                logger.debug(f"[DNABERT2v2] Switched attention to PyTorch native SDPA in {filepath}")
        except Exception:
            pass

    # Patch flash_attn_triton.py if used
    pattern = os.path.join(hf_cache_dir, "**", "flash_attn_triton.py")
    for filepath in glob.glob(pattern, recursive=True):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

            replacements = {
                "tl.dot(q, k, trans_b=True)": "tl.dot(q, tl.trans(k))",
                "tl.dot(p.to(do.dtype), do, trans_a=True)": "tl.dot(tl.trans(p.to(do.dtype)), do)",
                "tl.dot(do, v, trans_b=True)": "tl.dot(do, tl.trans(v))",
                "tl.dot(ds, q, trans_a=True)": "tl.dot(tl.trans(ds), q)",
            }
            modified = False
            for old, new in replacements.items():
                if old in content:
                    content = content.replace(old, new)
                    modified = True

            if modified:
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(content)
                logger.debug(f"[DNABERT2v2] Automatically patched Triton syntax in {filepath}")
        except Exception:
            pass


class DNABERT2v2(AbstractModel):
    """
    DNABERT2v2 model for DNA sequence embedding computation.
    Loads 'zhihan1996/DNABERT-2-117M' directly from Hugging Face Hub.
    """

    MODEL_NAME = "zhihan1996/DNABERT-2-117M"

    def __init__(
        self,
        merging_strategy: AbstractMergerStrategy = TruncateStrategy(),
        device: str = "cpu",
        overlap: int = 0,
        max_seq_len: int = 10000,
        load_model: bool = True,
    ) -> None:

        super().__init__(
            max_seq_len=max_seq_len,
            merging_strategy=merging_strategy,
            device=device,
            overlap=overlap,
            load_model=load_model,
            batch_size=1,
        )

        if self.load_model:
            # Silence internal transformers logging messages during load
            current_log_level = logging.root.level
            Logging.set_logging_level()

            logger.debug(f"[DNABERT2v2] Loading model weights from Hugging Face: {self.MODEL_NAME}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.MODEL_NAME, trust_remote_code=True
            )
            config = BertConfig.from_pretrained(self.MODEL_NAME)
            self.model = AutoModel.from_pretrained(
                self.MODEL_NAME, trust_remote_code=True, config=config
            )
            _patch_triton_syntax_if_needed()

            Logging.set_logging_level(current_log_level)

            self.model.to(self.device)
            self.model.eval()

        logger.debug(f"[DNABERT2v2] Max sequence length for DNABERT2v2: {self.max_seq_len}")

    def _compute_single_embedding(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = tokens.to(self.device)
        attention_mask = (tokens != self.tokenizer.pad_token_id).to(self.device)

        with torch.no_grad():
            output = self.model(tokens, attention_mask=attention_mask)[0]  # [B, L, H]

        attention_mask_unsq = torch.unsqueeze(attention_mask, dim=-1).to(output.dtype)

        # Mask-weighted mean pooling across sequence tokens
        mean_embed = (output * attention_mask_unsq).sum(dim=1) / attention_mask_unsq.sum(
            dim=1
        ).clamp(min=1e-9)

        return mean_embed

    def _encode(self, dna_sequences: list[str]) -> torch.Tensor:
        return self.tokenizer.batch_encode_plus(
            dna_sequences, return_tensors="pt", padding=True, truncation=True, max_length=8192
        )["input_ids"]

    def raw_name(self) -> str:
        return f"DNABERT2v2-RAW-ov{self.overlap}-maxlen{self.max_seq_len}"

    def name(self) -> str:
        return f"DNABERT2v2-{self.merging_strategy.name()}-ov{self.overlap}-maxlen{self.max_seq_len}"

    def __repr__(self) -> str:
        return f"DNABERT2v2(merging_strategy={self.merging_strategy}, overlap={self.overlap}, max_seq_len={self.max_seq_len})"
