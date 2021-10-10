import csv
from pprint import pprint
from typing import Dict, List

import pkg_resources
from t5.data.glue_utils import get_glue_metric, get_super_glue_metric
from t5.evaluation.metrics import accuracy, mean_multiclass_f1, rouge


SAFE_EXCLUDE_CRETERIA = [
    "template_bug",
    "negated_answers",
    "counting",
    "answer_span_indices",
    "non_natural_language",
    "generative_non_true_implausible",
]

AGGRESSIVE_EXCLUDE_CRETERIA = [
    "generative_non_true_task",
    "nontrivial_choices_hidden",
    "awkward_phrasing",
    "ungrammatical",
] + SAFE_EXCLUDE_CRETERIA


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


def exclude_bad_prompts(prompt: Dict) -> bool:
    for criterion in SAFE_EXCLUDE_CRETERIA:  # or AGGRESSIVE_EXCLUDE_CRETERIA
        if prompt.get(criterion):
            return False
    return True


def load_annotated_prompts() -> List[Dict]:
    annotated_csv_path = pkg_resources.resource_filename(__name__, "experiment_D3.csv")
    with open(annotated_csv_path) as in_file:
        reader = csv.DictReader(in_file)
        all_tasks = [row for row in reader]

    clean_tasks = list(filter(exclude_bad_prompts, all_tasks))

    # Assign metrics
    non_glue_eval_sets = list(NON_GLUE_METRICS.keys())
    for task in clean_tasks:
        if not task["do_eval"]:
            continue

        full_name = task["dataset_subset_template"]
        if full_name.startswith("glue"):
            subset = full_name.split("_")[1]
            task["metrics"] = get_glue_metric(subset)
        elif full_name.startswith("super_glue"):
            subset = full_name.split("_")[2]
            if subset in ("wsc.fixed", "multirc"):
                # TODO: WSC and MultiRC need special pre/postprocesing
                task["metrics"] = [accuracy]
                continue
            task["metrics"] = get_super_glue_metric(subset)

        for dataset_name in non_glue_eval_sets:
            if full_name.startswith(dataset_name):
                task["metrics"] = NON_GLUE_METRICS[dataset_name]

        # Skip rank_classification for now until we actually support it
        # if task["nontrivial_choices_hidden"]:
        #     # Trick of plugging in answer options and rank LM probabilites as predictions.
        #     # Required for all prompts with non_trivial_choices_hidden,
        #     # but could be used for other tasks as well where answer choices are given.
        #     if "metrics" not in task:
        #         task["metrics"] = [rank_classification]
        #     elif rank_classification not in task["metrics"]:
        #         task["metrics"].append(rank_classification)

        # should be already handled by NON_GLUE_METRICS
        # if task['generative_true_task'] or task['generative_non_true_task']:
        #     task['metrics'] = rouge

    return clean_tasks


def preview() -> None:
    clean_tasks = load_annotated_prompts()

    train_tasks = [t for t in clean_tasks if not t["skip_train"]]
    eval_tasks = [t for t in clean_tasks if t["do_eval"]]

    pprint([t["dataset_subset_template"] for t in train_tasks])
    print(len(train_tasks))

    pprint([f'{t["dataset_subset_template"]} {t["metrics"]}' for t in eval_tasks])
    print(len(eval_tasks))


if __name__ == "__main__":
    preview()
