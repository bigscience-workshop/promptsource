import csv
from typing import List, Tuple, Optional

import pkg_resources

from promptsource.templates import TemplateCollection


# from rich import inspect
from rich.pretty import pprint


def preview() -> None:
    experiment_path = pkg_resources.resource_filename(__name__, "experiment_D4.csv")
    d4_train: List[Tuple[str, Optional[str]]] = []
    d4_eval: List[Tuple[str, Optional[str]]] = []
    d3_train_gpt: List[Tuple[str, Optional[str]]] = []
    d3_train_sglue: List[Tuple[str, Optional[str]]] = []
    experiment_path = pkg_resources.resource_filename(__name__, "experiment_D4.csv")
    with open(experiment_path) as exp_file:
        reader = csv.DictReader(exp_file)
        for row in reader:
            if row["skip"]:
                continue
            if row["subset"] == "":
                row["subset"] = None  # to match promptsource.Template object
            dataset_subset = (row["HF_name"], row["subset"])
            if row["do_train"] == "TRUE":
                d4_train.append(dataset_subset)
            if row["do_eval"] == "TRUE":
                d4_eval.append(dataset_subset)
            if row["D3_do_train"] == "TRUE" and 'GPT' in row["seed_paper"]:
                d3_train_gpt.append(dataset_subset)
            if row["D3_do_train"] == "TRUE" and row["HF_name"] == 'super_glue':
                d3_train_sglue.append(dataset_subset)
    all_datasets = d4_train + d4_eval + d3_train_gpt + d3_train_sglue
    print(f'Number of non-desk-rejected datasets = {len(all_datasets)}')

    # print(f"Number of training sets = {len(train_sets)}")
    # print(f"Number of evaluation sets = {len(eval_sets)}")

    template_collection = TemplateCollection()
    missing_og_flags = []
    for dataset_name, subset_name in template_collection.keys:
        if (dataset_name, subset_name) not in d4_train:
            template_collection.remove(dataset_name, subset_name)
            continue
        OG = 0
        non_OG = 0
        dataset = template_collection.get_dataset(dataset_name, subset_name)
        for template_name in dataset.all_template_names:
            template = dataset[template_name]
            # if dataset_name == 'ropes':
            #     inspect(template.metadata)
            if template.metadata.original_task is True:
                OG += 1
            elif template.metadata.original_task is False:
                non_OG += 1
            elif template.metadata.original_task is None:
                missing_og_flags.append(dataset_name + subset_name + template_name)
                continue
        print(dataset_name, subset_name, OG, non_OG, sep='\t\t')
    print(len(template_collection))

    print('Missing original task flags:')
    pprint(missing_og_flags)


    # # print(d4_train_mixture)
    # print(f"Number of training templates = {len(d4_train_mixture)}")
    # # print(d4_eval_mixture)
    # print(f"Number of evaluation templates = {len(d4_eval_mixture)}")
    # # for i in seqio.TaskRegistry.names():
    # #     print(i)
    # print(f"Number of SeqIO registered templates = {len(seqio.TaskRegistry.names())}")
    # print("^ includes non-original task templates which are excluded from the eval mixture")


if __name__ == "__main__":
    preview()
