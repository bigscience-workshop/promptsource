import csv
from pprint import pprint


safe_creteria = [
    "template_bug",
    "negated_answers",
    "counting",
    "answer_span_indices",
    "non_natural_language",
    "generative_non_true_implausible",
]

aggressive_creteria = [
    "generative_non_true_task",
    "nontrivial_choices_hidden",
    "awkward_phrasing",
    "ungrammatical",
] + safe_creteria


def clean(prompt):
    for criterion in safe_creteria:  # or aggressive_creteria
        if prompt.get(criterion):
            return False
    return True


if __name__ == "__main__":
    with open("dataset_subset_template.csv") as in_file:
        reader = csv.DictReader(in_file)
        all_tasks = [row for row in reader]

    clean_tasks = list(filter(clean, all_tasks))

    train_tasks = [t for t in clean_tasks if not t["skip_train"]]
    eval_tasks = [t for t in clean_tasks if t["do_eval"]]

    pprint([t["dataset_subset_template"] for t in train_tasks])
    print(len(train_tasks))

    pprint([t["dataset_subset_template"] for t in eval_tasks])
    print(len(eval_tasks))
