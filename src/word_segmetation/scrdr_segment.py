from underthesea import word_tokenize


def segment_text(text: str) -> list[str]:
    """Tách từ tiếng Việt bằng Underthesea."""
    if not isinstance(text, str) or not text.strip():
        return []

    return word_tokenize(text)