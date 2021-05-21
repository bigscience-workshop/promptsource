"""A Draft of data augmentation for prompt/output instance.

This simple and stupid draft is meant to be a pot of stone soup.

Example:
    `augment("Business")` returns `"Enterprises"`

Note:
    The back-translation strategy uses two different models for increasing the diversity.
    The risk of distortion is as the above example shown.

Todo:
    * Speed and memory-footprint
    * Refactoring for a better implementation of Strategy pattern
    * API for batch processing
    * Special tokens that indicate augmentation invariants
"""
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline


en_fr_translator = pipeline("translation_en_to_fr")
fr_en_model_name = "Helsinki-NLP/opus-mt-ROMANCE-en"
fr_en_translator = pipeline(
    model=AutoModelForSeq2SeqLM.from_pretrained(fr_en_model_name),
    tokenizer=AutoTokenizer.from_pretrained(fr_en_model_name),
    task="translation_fr_to_en",
)


def _back_translation_strategy(text: str) -> str:
    return fr_en_translator(en_fr_translator(text)[0]["translation_text"])[0]["translation_text"]


def _noop_strategy(text: str) -> str:
    return text


augment = _back_translation_strategy
