"""
=============================================================================
backend/utils/text_utils.py — Text Preprocessing Utilities
=============================================================================

Provides clean, consistent text preprocessing for all NLP models in the pipeline.

Why preprocessing matters:
    Raw user input from a chat interface contains noise that can degrade model
    performance: HTML entities, excessive whitespace, emoji sequences, URLs,
    and Unicode artifacts. Consistent preprocessing ensures that inference-time
    text matches the distribution seen during training.

Functions:
    clean_text()        — Main cleaning pipeline
    expand_contractions() — "I'm" → "I am" (improves tokenisation)
    remove_urls()       — Remove hyperlinks
    remove_html()       — Strip HTML tags (for imported content)
    normalize_whitespace() — Collapse runs of spaces/newlines
    truncate_text()     — Hard limit with word boundary awareness
    extract_keywords()  — Simple keyword extraction for conversation title
=============================================================================
"""

from __future__ import annotations

import re
import unicodedata
from typing import List, Optional


# ---------------------------------------------------------------------------
# Contraction expansion map
# ---------------------------------------------------------------------------
CONTRACTIONS: dict[str, str] = {
    "i'm": "i am", "i've": "i have", "i'll": "i will", "i'd": "i would",
    "you're": "you are", "you've": "you have", "you'll": "you will", "you'd": "you would",
    "he's": "he is", "she's": "she is", "it's": "it is",
    "we're": "we are", "we've": "we have", "we'll": "we will", "we'd": "we would",
    "they're": "they are", "they've": "they have", "they'll": "they will", "they'd": "they would",
    "that's": "that is", "that'll": "that will",
    "there's": "there is", "there've": "there have",
    "who's": "who is", "who've": "who have", "who'll": "who will",
    "what's": "what is", "what'll": "what will",
    "can't": "cannot", "won't": "will not", "don't": "do not",
    "doesn't": "does not", "didn't": "did not", "isn't": "is not",
    "aren't": "are not", "wasn't": "was not", "weren't": "were not",
    "haven't": "have not", "hasn't": "has not", "hadn't": "had not",
    "wouldn't": "would not", "shouldn't": "should not", "couldn't": "could not",
    "mustn't": "must not", "needn't": "need not",
    "let's": "let us", "that'd": "that would",
}

# Compile pattern once for efficiency
_CONTRACTION_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in CONTRACTIONS) + r")\b",
    re.IGNORECASE
)


def expand_contractions(text: str) -> str:
    """
    Expand English contractions.

    "I'm feeling anxious" → "I am feeling anxious"

    Why: Transformers' WordPiece tokeniser sometimes splits "I'm" into
    ["i", "'", "m"] which can fragment the meaning. Expanding first
    ensures cleaner tokenisation.
    """
    def replace(match: re.Match) -> str:
        matched = match.group(0)
        expanded = CONTRACTIONS.get(matched.lower(), matched)
        # Preserve original capitalisation of first character
        if matched[0].isupper():
            expanded = expanded[0].upper() + expanded[1:]
        return expanded

    return _CONTRACTION_PATTERN.sub(replace, text)


def remove_urls(text: str) -> str:
    """Remove HTTP/HTTPS URLs from text."""
    url_pattern = re.compile(
        r"https?://\S+|www\.\S+",
        re.IGNORECASE
    )
    return url_pattern.sub("[URL]", text)


def remove_html(text: str) -> str:
    """Strip HTML tags using regex (lightweight — no BS4 needed for short inputs)."""
    html_pattern = re.compile(r"<[^>]+>")
    text = html_pattern.sub(" ", text)
    # Decode common HTML entities
    entities = {"&amp;": "&", "&lt;": "<", "&gt;": ">", "&quot;": '"', "&#39;": "'", "&nbsp;": " "}
    for entity, char in entities.items():
        text = text.replace(entity, char)
    return text


def normalize_whitespace(text: str) -> str:
    """Collapse multiple spaces/newlines into a single space."""
    text = re.sub(r"[\n\r\t]+", " ", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def normalize_unicode(text: str) -> str:
    """
    Normalize Unicode to NFC form and optionally remove exotic characters.

    NFC (Canonical Decomposition, followed by Canonical Composition) is the
    standard form used by most NLP libraries. It ensures that equivalent
    Unicode sequences (e.g., é as U+00E9 vs e + U+0301) are treated identically.
    """
    return unicodedata.normalize("NFC", text)


def clean_text(
    text: str,
    expand_contraction: bool = True,
    remove_url: bool = True,
    remove_html_tags: bool = True,
    lowercase: bool = False,
    max_length: Optional[int] = None,
) -> str:
    """
    Main text cleaning pipeline.

    Processing order matters:
        1. Unicode normalisation (must be first)
        2. HTML removal (before URL detection, some HTML has URLs)
        3. URL removal
        4. Contraction expansion (before whitespace normalisation)
        5. Whitespace normalisation (last, catches artifacts from prior steps)

    Args:
        text: Raw input text
        expand_contraction: Whether to expand contractions
        remove_url: Whether to replace URLs with [URL]
        remove_html_tags: Whether to strip HTML
        lowercase: Whether to lowercase (False for generation input, True for classifiers)
        max_length: Hard character limit (word-boundary aware)

    Returns:
        Cleaned text string
    """
    if not text or not text.strip():
        return ""

    text = normalize_unicode(text)

    if remove_html_tags:
        text = remove_html(text)

    if remove_url:
        text = remove_urls(text)

    if expand_contraction:
        text = expand_contractions(text)

    text = normalize_whitespace(text)

    if lowercase:
        text = text.lower()

    if max_length and len(text) > max_length:
        text = truncate_text(text, max_length)

    return text


def truncate_text(text: str, max_chars: int) -> str:
    """
    Truncate text at a word boundary near max_chars.

    Avoids cutting in the middle of a word, which can confuse tokenisers.
    """
    if len(text) <= max_chars:
        return text
    # Find the last space before max_chars
    truncated = text[:max_chars]
    last_space = truncated.rfind(" ")
    if last_space > max_chars * 0.7:  # Don't truncate too aggressively
        return truncated[:last_space] + "…"
    return truncated + "…"


def extract_conversation_title(first_message: str, max_words: int = 6) -> str:
    """
    Generate a conversation title from the first user message.

    Uses the first N meaningful words, capitalised as a title.
    Filters out common filler words.

    e.g. "I've been feeling really anxious about my exams" → "Feeling Anxious About Exams"
    """
    if not first_message.strip():
        return "New Conversation"

    # Truncate at around 30 characters on a word boundary
    title = first_message.strip()
    if len(title) > 30:
        title = title[:30].rsplit(' ', 1)[0] + "..."
        
    return title


def format_context_prompt(
    conversation_history: List[dict],
    emotion: str,
    intent: str,
    sentiment: str,
    max_turns: int = 6,
) -> str:
    """
    Format conversation history + analysis into a FLAN-T5 prompt.

    FLAN-T5 is an instruction-tuned model — it works best when given
    an explicit instruction prefix followed by context.

    The prompt includes:
        - Task instruction (empathetic mental health support)
        - Detected emotion and intent (so the model conditions its response)
        - Last N conversation turns
        - "Response:" cue to trigger generation

    Args:
        conversation_history: List of {"role": ..., "content": ...} dicts
        emotion: Detected emotion label
        intent: Detected intent label
        sentiment: Detected sentiment label
        max_turns: Maximum turns to include (sliding window)

    Returns:
        Formatted prompt string
    """
    # Take the last max_turns messages
    recent_history = conversation_history[-max_turns:]

    # For the base (untrained) FLAN-T5-small model, long preambles cause it to
    # just autocomplete the persona (e.g. "You're a compassionate person.").
    # We use a much simpler direct instruction here to get reasonable zero-shot responses.
    
    last_msg = ""
    for turn in reversed(recent_history):
        if turn["role"] == "user":
            last_msg = turn["content"]
            break

    prompt = (
        f"Respond empathetically and supportively to the user.\n"
        f"User: {last_msg}\n"
        f"Response:"
    )

    return prompt
