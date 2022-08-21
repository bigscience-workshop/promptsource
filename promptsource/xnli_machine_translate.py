import os
import re

from promptsource.templates import Template, TemplateCollection


PROMPTS = [
    "GPT-3 style",
    "can we infer",
    "justified in saying",
    "guaranteed/possible/impossible",
    "MNLI crowdsource",
]

LANGS = [
    "ar",
    "es",
    "fr",
    "hi",
    "sw",
    "ur",
    "vi",
    "zh",
]
# Path to key
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/niklasmuennighoff/Desktop/gcp_translate_key.json"


def translate(target, text):
    """Translates text into the target language.
    Target must be an ISO 639-1 language code.
    See https://g.co/cloud/translate/v2/translate-reference#supported_languages
    (pip install --upgrade google-api-python-client)
    pip install google-cloud-translate
    """
    import six
    from google.cloud import translate_v2 as translate

    translate_client = translate.Client()
    if isinstance(text, six.binary_type):
        text = text.decode("utf-8")
    # Text can also be a sequence of strings, in which case this method
    # will return a sequence of results for each text.
    result = translate_client.translate(text, target_language=target)
    print("Text: {}".format(result["input"]))
    print("Translation: {}".format(result["translatedText"]))
    print("Detected source language: {}".format(result["detectedSourceLanguage"]))
    return result["translatedText"]


def normalize_string(zh_string, en_string):
    """
    This is not specific to zh just to given an example.
    Replaces the content in brackets in zh_string with the content in brackets from en_string.
    All else is left the same in zh_string.
    Args:
        zh_string: {{前提}} 问题：{{假设}} 对、错或两者都不是？ ||| {{ answer_choices[标签] }}
        en_string: {{premise}} Question: {{hypothesis}} True, False, or Neither? ||| {{ answer_choices[label] }}
    Returns:
        zh_string_normalized: {{premise}} 问题：{{hypothesis}} 对、错或两者都不是？ ||| {{ answer_choices[label] }}
    """
    zh_string_normalized = zh_string
    en_string_normalized = en_string
    # Find all the content in brackets in zh_string
    zh_bracket_content = re.findall(r"{{(.*?)}}", zh_string)
    # Find all the content in brackets in en_string
    en_bracket_content = re.findall(r"{{(.*?)}}", en_string)
    # Replace the content in brackets in zh_string with the content in brackets from en_string
    for i in range(len(zh_bracket_content)):
        zh_string_normalized = zh_string_normalized.replace(zh_bracket_content[i], en_bracket_content[i])
    return zh_string_normalized


template_collection = TemplateCollection()
source_templates = template_collection.get_dataset("xnli", "en")

for lang in LANGS:
    target_templates = template_collection.get_dataset("xnli", lang)
    for uid, template in source_templates.templates.items():
        if template.name.strip() not in PROMPTS:
            continue
        print(f"Translating {template.name.strip()} to {lang}")
        answer_choices = []
        if template.answer_choices is not None:
            choices = template.answer_choices.split("|||")
            for c in choices:
                answer_choices.append(translate(lang, c.strip()))
        or_jinja = template.jinja.strip()
        jinja = normalize_string(translate(lang, or_jinja), or_jinja)
        template_name = template.name.strip() + f"_{lang}mt"
        target_template = Template(
            template_name, jinja=jinja, reference="", answer_choices=" ||| ".join(answer_choices)
        )
        target_templates.add_template(target_template)
