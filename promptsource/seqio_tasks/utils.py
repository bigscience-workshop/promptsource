import re

import datasets
import tensorflow as tf

import promptsource.utils


def feature_to_spec(feature, length=False):
    if isinstance(feature, datasets.ClassLabel):
        return tf.TensorSpec(shape=() if not length else (None if length == -1 else length,), dtype=tf.int64)
    elif isinstance(feature, datasets.Value):
        return tf.TensorSpec(
            shape=() if not length else (None if length == -1 else length,), dtype=getattr(tf.dtypes, feature.dtype)
        )
    elif hasattr(feature, "dtype") and hasattr(feature, "shape"):
        return tf.TensorSpec(shape=feature.shape, dtype=feature.dtype)
    elif isinstance(feature, datasets.Sequence):
        return feature_to_spec(feature.feature, length=feature.length)
    elif isinstance(feature, list):
        return [feature_to_spec(f, length=length) for f in feature]
    elif isinstance(feature, dict):
        return {k: feature_to_spec(v, length=length) for k, v in feature.items()}
    else:
        raise ValueError(f"Unparseable feature type {type(feature)}")


def hf_dataset_to_tf_dataset(dataset):
    return tf.data.Dataset.from_generator(
        dataset.__iter__, output_signature={k: feature_to_spec(v) for k, v in dataset.features.items()}
    )


def apply_template(dataset, template):
    def map_fn(ex):
        ex = promptsource.utils.removeHyphen(ex)
        inputs_and_targets = template.apply(ex)
        answer_choices = template.get_answer_choices_list(ex)
        if len(inputs_and_targets) == 2:
            inputs, targets = inputs_and_targets
            if targets == "":
                ex = {"inputs": inputs, "targets": "<NO LABEL>"}
            else:
                ex = {"inputs": inputs, "targets": targets}
        # When template results in an empty example, template.apply returns [""]
        # Also, if the template gets split wrong, len can be > 2
        # We will filter these out later
        else:
            ex = {"inputs": "", "targets": ""}

        if answer_choices:
            ex["answer_choices"] = answer_choices

        return ex

    def filter_fn(ex):
        return len(ex["inputs"]) > 0 and len(ex["targets"]) > 0

    original_columns = dataset.column_names
    dataset = dataset.map(map_fn).filter(filter_fn)
    # map keeps original columns, remove them
    return dataset.remove_columns(set(original_columns) - {"inputs", "targets", "answer_choices"})


def get_dataset_splits(dataset_name, subset_name=None):
    info = datasets.get_dataset_infos(dataset_name)
    subset_name = subset_name or list(info.keys())[0]
    return info[subset_name].splits


def task_clean(text):
    # Clean the text according to allowed characters for a task name
    return re.sub(r"[^\w\d\._]+", "_", text)


def get_task_name(dataset_name, subset_name, template_name):
    return task_clean(dataset_name + (f"_{subset_name}_" if subset_name is not None else "_") + template_name)
