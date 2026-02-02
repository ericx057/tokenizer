"""
Ruler - Tokenizer Alignment for Llama 3.1.

Ensures strict token counting using the exact Llama 3.1 tokenizer configuration.
Uses tokenizer.apply_chat_template() for accurate prompt building.
Includes LRU caching to reduce redundant tokenization.
"""

from functools import lru_cache
from typing import List, Dict, Any, Tuple
from transformers import AutoTokenizer


class TokenizerMismatchError(Exception):
    """Raised when tokenizer vocab_size does not match expected Llama 3.1 config."""

    pass


class Ruler:
    """
    Handles strict tokenization aligned with Llama 3.1. 

    All chunking logic operates on token IDs, never on string length.
    Uses the tokenizer's native chat template for accurate token counting.
    Includes LRU caching to avoid redundant tokenization.
    
    Cache Scope:
        LRU caches are INSTANCE-LEVEL. Multiple Engine instances do NOT share
        caches even if using the same tokenizer model. Each Ruler has its own
        independent cache. Call clear_cache() on each instance independently.
        
        Memory: 256 tokenize entries + 512 count entries per instance.
        For typical chunks of ~4KB, worst case ~3MB memory per instance.
    """

    def __init__(self, model_name: str = "meta-llama/Llama-3.1-8B-Instruct", expected_vocab_size: int = 128256):
        """
        Initialize the Ruler with a Llama 3.1 compatible tokenizer.

        Args:
            model_name: HuggingFace model identifier for tokenizer.
            expected_vocab_size: Expected vocab size (128256 for Llama 3.1).

        Raises:
            TokenizerMismatchError: If vocab_size does not match expected.
        """
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        actual_vocab_size = self._tokenizer.vocab_size

        if actual_vocab_size != expected_vocab_size:
            raise TokenizerMismatchError(
                f"Expected vocab_size={expected_vocab_size}, got {actual_vocab_size}. "
                f"Ensure you are using a Llama 3.1 compatible tokenizer."
            )
        
        # Create cached versions of expensive operations
        # Using tuple return for hashability in cache
        # Memory note: 256 tokenize entries + 512 count entries
        # For typical chunks of ~4KB, worst case ~3MB memory
        # Call clear_cache() between independent requests if memory is constrained
        self._cached_tokenize = lru_cache(maxsize=256)(self._tokenize_impl)
        self._cached_count = lru_cache(maxsize=512)(self._count_impl)

    def _tokenize_impl(self, text: str) -> Tuple[int, ...]:
        """Internal tokenization, returns tuple for caching."""
        return tuple(self._tokenizer.encode(text, add_special_tokens=False))

    def _count_impl(self, text: str) -> int:
        """Internal count, cacheable."""
        return len(self._tokenize_impl(text))

    def tokenize(self, text: str) -> List[int]:
        """
        Convert text to token IDs (cached).

        Args:
            text: Raw input string.

        Returns:
            List of token IDs.
        """
        return list(self._cached_tokenize(text))

    def detokenize(self, ids: List[int]) -> str:
        """
        Convert token IDs back to text.

        Args:
            ids: List of token IDs.

        Returns:
            Decoded string.
        """
        return self._tokenizer.decode(ids)

    def count(self, text: str) -> int:
        """
        Count tokens in raw text WITHOUT special tokens (cached).

        Use for: measuring content size, chunk size calculations.
        Do NOT use for: measuring actual prompt size sent to model.
        
        For prompt-aware counting that includes special tokens (e.g.,
        <|begin_of_text|>, <|start_header_id|>), use count_prompt_tokens().

        Args:
            text: Raw input string.

        Returns:
            Number of tokens (excluding special tokens).
        """
        return self._cached_count(text)

    def clear_cache(self) -> None:
        """Clear tokenization cache.
        
        Call between requests if memory usage is a concern.
        """
        self._cached_tokenize.cache_clear()
        self._cached_count.cache_clear()

    def cache_info(self) -> Dict[str, Any]:
        """Get cache statistics."""
        tokenize_info = self._cached_tokenize.cache_info()
        count_info = self._cached_count.cache_info()
        return {
            "tokenize": {
                "hits": tokenize_info.hits,
                "misses": tokenize_info.misses,
                "size": tokenize_info.currsize,
            },
            "count": {
                "hits": count_info.hits,
                "misses": count_info.misses,
                "size": count_info.currsize,
            },
        }

    def build_prompt(
        self,
        system_content: str,
        user_content: str,
        add_generation_prompt: bool = True,
    ) -> str:
        """
        Build a prompt using the tokenizer's native chat template.

        This is the single source of truth for prompt formatting.

        Args:
            system_content: System message content.
            user_content: User message content.
            add_generation_prompt: Whether to add the assistant header.

        Returns:
            Formatted prompt string.
        """
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]
        return self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    def count_prompt_tokens(
        self,
        system_content: str,
        user_content: str,
        add_generation_prompt: bool = True,
    ) -> int:
        """
        Count tokens in a formatted prompt INCLUDING special tokens.

        Uses apply_chat_template for accurate counting that matches
        what will actually be sent to the model. This includes:
        - <|begin_of_text|>
        - <|start_header_id|>system<|end_header_id|>
        - Role headers and formatting tokens
        
        Use this for: determining if prompt fits in context window.
        Do NOT use count() for prompt size calculations.

        Args:
            system_content: System message content.
            user_content: User message content.
            add_generation_prompt: Whether to include assistant header tokens.

        Returns:
            Total token count including all special tokens.
        """
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]
        token_ids = self._tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
        )
        return len(token_ids)

    @property
    def vocab_size(self) -> int:
        """Return the tokenizer's vocabulary size."""
        return self._tokenizer.vocab_size
