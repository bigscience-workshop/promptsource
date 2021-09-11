import csv
from re import template
from typing import Dict, List
import pkg_resources

from rich import inspect
from rich.pretty import pprint
from t5.data.glue_utils import get_glue_metric, get_super_glue_metric
from t5.evaluation.metrics import accuracy, mean_multiclass_f1, rouge

from promptsource.templates import TemplateCollection


NON_GLUE_METRICS = {  # for those with do_eval = True
    "anli": [accuracy],
    "hans": [accuracy],
    "circa_goldstandard1_judgement": [mean_multiclass_f1(num_classes=8), accuracy],
    "circa_goldstandard2_judgement": [mean_multiclass_f1(num_classes=5), accuracy],
    "mc_taco": [accuracy],
    "nq_open": [accuracy],
    "qa_srl": [accuracy],
    "openbookqa": [accuracy],
    "race": [accuracy],
    "social_i_qa": [accuracy],
    "emo": [mean_multiclass_f1(num_classes=4)],
    "xsum": [rouge],
}


def preview() -> None:
    experiment_path = pkg_resources.resource_filename(__name__, "experiment_D4.csv")

    train_sets = []
    eval_sets = []
    train_set_names = []
    eval_set_names = []
    with open(experiment_path) as exp_file:
        reader = csv.DictReader(exp_file)
        for row in reader:
            if row["skip"]:
                continue
            if row["subset"] == "":
                row["subset"] = None  # to match promptsource.Template object
            if row["do_train"] == "TRUE":
                train_sets.append(row)
                train_set_names.append((row["HF_name"], row["subset"]))
            if row["do_eval"] == "TRUE":
                eval_sets.append(row)
                eval_set_names.append((row["HF_name"], row["subset"]))

        # print(f'Number of non-desk-rejected datasets = {len(all_datasets)}')

    D4_names = train_set_names + eval_set_names
    print(f"Number of training sets = {len(train_sets)}")
    print(f"Number of evaluation sets = {len(eval_sets)}")

    template_collection = TemplateCollection()
    print(type(template_collection))
    for dataset_name, subset_name in template_collection.keys:
        if (dataset_name, subset_name) not in train_set_names:  # D4_names:
            template_collection.remove(dataset_name, subset_name)
            continue
        OG = 0
        non_OG = 0
        dataset = template_collection.get_dataset(dataset_name, subset_name)
        for template_name in dataset.all_template_names:
            template = dataset[template_name]
            # if dataset_name == 'ropes':
            #     inspect(template.metadata)
        #     if template.metadata.original_task is True:
        #         OG += 1
        #     elif template.metadata.original_task is False:
        #         non_OG += 1
        #     elif template.metadata.original_task is None:
        #         print(dataset_name, 'has not flagged original tasks')
        #         continue
        # print(dataset_name, subset_name, OG, non_OG)
    print(len(template_collection))


if __name__ == "__main__":
    preview()
