# coding=utf-8

import datasets
import pkg_resources
import requests

from promptsource.templates import INCLUDED_USERS


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


def get_dataset_builder(path, conf=None):
    "Get a dataset builder from name and conf."
    # `datasets.load.prepare_module` pulls infos from hf/datasets's master.
    # story_cloze hasn't been merged yet (https://github.com/huggingface/datasets/pull/2907)
    # This is a temporary fix for the tests (more specifically test_templates.py)
    # Once PR 2907 is merged, we can remove this if condition (along with the `custom_datasets` folder)
    # Also see `promptsource.seqio_tasks.utils.get_dataset_splits`
    if path == "story_cloze":
        path = pkg_resources.resource_filename("promptsource", "custom_datasets/story_cloze")
    module_path = datasets.load.prepare_module(path, dataset=True)
    builder_cls = datasets.load.import_main_class(module_path[0], dataset=True)
    if conf:
        builder_instance = builder_cls(name=conf, cache_dir=None, hash=module_path[1])
    else:
        builder_instance = builder_cls(cache_dir=None, hash=module_path[1])
    return builder_instance


def get_dataset(path, conf=None):
    "Get a dataset from name and conf."
    builder_instance = get_dataset_builder(path, conf)
    fail = False
    if builder_instance.manual_download_instructions is None and builder_instance.info.size_in_bytes is not None:
        builder_instance.download_and_prepare()
        dts = builder_instance.as_dataset()
        dataset = dts
    else:
        try:
            dataset = datasets.load_dataset(path, conf)
        except:
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
    """
    Filter English datasets based on language tags in metadata.

    Also includes the datasets of any users listed in INCLUDED_USERS
    """
    english_datasets = []

    response = requests.get("https://huggingface.co/api/datasets?full=true")
    tags = response.json()

    for dataset in tags:
        dataset_name = dataset["id"]

        is_community_dataset = "/" in dataset_name
        if is_community_dataset:
            user = dataset_name.split("/")[0]
            if user in INCLUDED_USERS:
                english_datasets.append(dataset_name)
            continue

        if "card_data" not in dataset:
            continue
        metadata = dataset["card_data"]

        if "languages" not in metadata:
            continue
        languages = metadata["languages"]

        if "en" in languages or "en-US" in languages:
            english_datasets.append(dataset_name)

    return sorted(english_datasets)


def list_datasets(template_collection, _state):
    """Get all the datasets to work with."""
    dataset_list = filter_english_datasets()
    dataset_list.sort(key=lambda x: x.lower())
    return dataset_list
