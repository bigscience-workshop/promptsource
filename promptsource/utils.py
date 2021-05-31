# coding=utf-8

import datasets
import requests


# Hard-coded additional English datasets
# These datasets have either metadata missing or language informations missing
# so we manually identified them.
_ADDITIONAL_ENGLISH_DATSETS = [
    "aeslc",
    "ai2_arc",
    "amazon_us_reviews",
    "anli",
    "art",
    "aslg_pc12",
    "asnq",
    "biomrc",
    "blended_skill_talk",
    "blimp",
    "blog_authorship_corpus",
    "bookcorpus",
    "bookcorpusopen",
    "boolq",
    "break_data",
    "cfq",
    "civil_comments",
    "com_qa",
    "common_gen",
    "commonsense_qa",
    "conll2000",
    "coqa",
    "cornell_movie_dialog",
    "cos_e",
    "cosmos_qa",
    "crd3",
    "crime_and_punish",
    "daily_dialog",
    "definite_pronoun_resolution",
    "discofuse",
    "docred",
    "doqa",
    "drop",
    "emo",
    "emotion",
    "empathetic_dialogues",
    "eraser_multi_rc",
    "esnli",
    "event2Mind",
    "fever",
    "gap",
    "gigaword",
    "guardian_authorship",
    "hans",
    "hellaswag",
    "hotpot_qa",
    "hyperpartisan_news_detection",
    "imdb",
    "jeopardy",
    "lc_quad",
    "math_dataset",
    "math_qa",
    "mlqa",
    "movie_rationales",
    "ms_marco",
    "multi_news",
    "mwsc",
    "natural_questions",
    "newsgroup",
    "newsroom",
    "openbookqa",
    "openwebtext",
    "opinosis",
    "pg19",
    "qa_zre",
    "qangaroo",
    "qanta",
    "qasc",
    "quail",
    "quarel",
    "quartz",
    "quora",
    "quoref",
    "race",
    "reddit",
    "reddit_tifu",
    "reuters21578",
    "rotten_tomatoes",
    "scan",
    "scicite",
    "scientific_papers",
    "scifact",
    "sciq",
    "scitail",
    "search_qa",
    "sem_eval_2010_task_8",
    "sentiment140",
    "social_i_qa",
    "squad_v2",
    "squadshifts",
    "super_glue",
    "trec",
    "trivia_qa",
    "tydiqa",
    "ubuntu_dialogs_corpus",
    "web_nlg",
    "web_of_science",
    "web_questions",
    "wiki40b",
    "wiki_qa",
    "wiki_snippets",
    "wiki_split",
    "wikipedia",
    "wikisql",
    "wikitext",
    "winogrande",
    "wiqa",
    "wnut_17",
    "xnli",
    "xquad",
    "xsum",
    "xtreme",
    "yelp_polarity",
]


def removeHyphen(example):
    example_clean = {}
    for key in example.keys():
        if "-" in key:
            new_key = key.replace("-", "_")
            example_clean[new_key] = example[key]
        else:
            example_clean[key] = example[key]
    example = example_clean
    return example


def renameDatasetColumn(dataset):
    col_names = dataset.column_names
    for cols in col_names:
        if "-" in cols:
            dataset = dataset.rename_column(cols, cols.replace("-", "_"))
    return dataset


#
# Helper functions for datasets library
#


def get_dataset(path, conf=None):
    "Get a dataset from name and conf."
    module_path = datasets.load.prepare_module(path, dataset=True)
    builder_cls = datasets.load.import_main_class(module_path[0], dataset=True)
    if conf:
        builder_instance = builder_cls(name=conf, cache_dir=None)
    else:
        builder_instance = builder_cls(cache_dir=None)
    fail = False
    if builder_instance.manual_download_instructions is None and builder_instance.info.size_in_bytes is not None:
        builder_instance.download_and_prepare()
        dts = builder_instance.as_dataset()
        dataset = dts
    else:
        dataset = builder_instance
        fail = True
    return dataset, fail


def get_dataset_confs(path):
    "Get the list of confs for a dataset."
    module_path = datasets.load.prepare_module(path, dataset=True)
    # Get dataset builder class from the processing script
    builder_cls = datasets.load.import_main_class(module_path[0], dataset=True)
    # Instantiate the dataset builder
    confs = builder_cls.BUILDER_CONFIGS
    if confs and len(confs) > 1:
        return confs
    return []


def render_features(features):
    """Recursively render the dataset schema (i.e. the fields)."""
    if isinstance(features, dict):
        return {k: render_features(v) for k, v in features.items()}
    if isinstance(features, datasets.features.ClassLabel):
        return features.names

    if isinstance(features, datasets.features.Value):
        return features.dtype

    if isinstance(features, datasets.features.Sequence):
        return {"[]": render_features(features.feature)}
    return features


#
# Loads dataset information
#


def filter_english_datasets():
    """Filter English datasets based on language tags in metadata"""
    english_datasets = []

    response = requests.get("https://huggingface.co/api/datasets?full=true")
    tags = response.json()

    for dataset in tags:
        dataset_name = dataset["id"]

        is_community_dataset = "/" in dataset_name
        if is_community_dataset:
            continue

        if "card_data" not in dataset:
            continue
        metadata = dataset["card_data"]

        if "languages" not in metadata:
            continue
        languages = metadata["languages"]

        if "en" in languages:
            english_datasets.append(dataset_name)

    all_english_datasets = list(set(english_datasets + _ADDITIONAL_ENGLISH_DATSETS))
    return sorted(all_english_datasets)


def list_datasets(template_collection, _priority_filter, _priority_max_templates, _state):
    """Get all the datasets to work with."""
    dataset_list = filter_english_datasets()
    count_dict = template_collection.get_templates_count()
    if _priority_filter:
        dataset_list = list(
            set(dataset_list)
            - set(
                list(
                    d
                    for d in count_dict
                    if count_dict[d] > _priority_max_templates and d != _state.working_priority_ds
                )
            )
        )
        dataset_list.sort()
    return dataset_list
