"""Token counting utility using tiktoken."""

from __future__ import annotations


def count_tokens(text: str, model: str = "cl100k_base") -> int:
    """Estimate the number of tokens in *text*.

    Args:
        text: The input text.
        model: tiktoken encoding name. Defaults to ``cl100k_base``.

    Returns:
        Estimated token count.
    """
    try:
        import tiktoken
        encoding = tiktoken.get_encoding(model)
        return len(encoding.encode(text))
    except Exception:
        # Fallback: approximate 1 token ≈ 4 characters for English
        return max(1, len(text) // 4)
