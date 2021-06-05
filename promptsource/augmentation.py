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
import re
from typing import Iterable, Mapping

from jinja2 import Environment
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline


jinja_env = Environment()
en_fr_translator = pipeline("translation_en_to_fr")
fr_en_model_name = "Helsinki-NLP/opus-mt-ROMANCE-en"
fr_en_translator = pipeline(
    model=AutoModelForSeq2SeqLM.from_pretrained(fr_en_model_name),
    tokenizer=AutoTokenizer.from_pretrained(fr_en_model_name),
    task="translation_fr_to_en",
)

CONST_REGEX = re.compile(r"(?P<cnst>{{\"[^\"]*\"}})")


def _decorate_constants(tmpl_rows: Iterable[str]) -> (Iterable[str], Mapping[str, str]):
    decorated_tmpl_rows = []
    cnst_dict = {}
    for i, tmpl_row in enumerate(tmpl_rows):
        decorated_tmpl_row = ""
        offset = 0
        for j, mo in enumerate(CONST_REGEX.finditer(tmpl_row)):
            # The format of special tokens below should be general enough, but it seems making translations worse.
            cnst_key = f"x_cnst_{i}_{j}"
            cnst_val = mo.group()
            cnst_dict[cnst_key] = cnst_val
            end, shift = mo.span()
            decorated_tmpl_row += f"{tmpl_row[offset:end]}{cnst_key}"
            offset = shift
        decorated_tmpl_row += tmpl_row[offset:]
        decorated_tmpl_rows.append(decorated_tmpl_row)
    return decorated_tmpl_rows, cnst_dict


def _decorate_text_var(tmpl_str: str) -> str:
    tmpl = jinja_env.from_string(tmpl_str)
    return tmpl.render(text="x_txt")


def _get_translation(item: Mapping[str, str]) -> str:
    return item["translation_text"]


def _restore_constants_and_text_var(decorated_text: str, cnst_dict: Mapping[str, str]) -> str:
    for k, v in cnst_dict.items():
        decorated_text = decorated_text.replace(k, v)
    return decorated_text.replace("x_txt", "{{text}}")


def back_translation_strategy(text: str) -> str:
    tmpl_str, fld_str = text.split(" |||")
    tmpl_rows = tmpl_str.splitlines()
    altered_tmpl_rows, cnst_map = _decorate_constants(tmpl_rows)
    altered_tmpl_rows = list(map(_decorate_text_var, altered_tmpl_rows))
    fr_altered_tmpl_rows = list(map(_get_translation, en_fr_translator(altered_tmpl_rows)))
    en_altered_tmpl_str = "\n".join(map(_get_translation, fr_en_translator(fr_altered_tmpl_rows)))
    en_restored_tmpl_str = _restore_constants_and_text_var(en_altered_tmpl_str, cnst_map)
    return en_restored_tmpl_str + " |||" + fld_str


def noop_strategy(text: str) -> str:
    return text


augment = back_translation_strategy
