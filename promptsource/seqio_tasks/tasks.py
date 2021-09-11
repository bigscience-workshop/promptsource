import functools
import re
import csv
import pkg_resources

import datasets
import seqio
import t5
import tensorflow as tf

import promptsource.templates

from . import utils


# Tasks deemed as clean/useful
# annotated_tasks = load_annotated_prompts.load_annotated_prompts()
# CLEAN_TASKS = [t["dataset_subset_template"] for t in annotated_tasks if not t["skip_train"]]
# CLEAN_EVAL_TASKS = [t["dataset_subset_template"] for t in annotated_tasks if t["do_eval"]]
# EVAL_METRICS = {t["dataset_subset_template"]: t["metrics"] for t in annotated_tasks if t["do_eval"]}


DATASET_BLACKLIST = [
    # Need to special-case ANLI due to weird split conventions
    ("anli", None),
]


def strip_whitespace(output_or_target, example=None, is_target=False):
    """Cached tasks from promptsource all have a leading space on the ground-truth targets."""
    return output_or_target.strip()


def get_label_strings(template):
    target = template.jinja.split("|||")[1]
    label_list_re = r"^([^\{\}]*)\{\{\s*(\[\s*[\"|\'].*[\"|\']\s*\])\s*\[.*\]\s*\}\}([^\{\}]*)$"
    label_string_match = re.search(label_list_re, target.strip())

    if label_string_match:
        before_label = label_string_match.group(1)
        labels = eval(label_string_match.group(2))
        after_label = label_string_match.group(3)
        labels = [before_label + label + after_label for label in labels]
        return labels


def maybe_get_class_id_postprocessor(template):
    labels = get_label_strings(template)
    if labels is not None:

        def postprocess_fn(output_or_target, example=None, is_target=False):
            output_or_target = strip_whitespace(output_or_target)
            return t5.data.postprocessors.string_label_to_class_id(output_or_target, label_classes=labels)

        return postprocess_fn

    else:
        return strip_whitespace


def get_tf_dataset(split, shuffle_files, seed, dataset_name, subset_name, template, split_mapping):
    # HF datasets does not support file-level shuffling
    del shuffle_files, seed
    dataset = datasets.load_dataset(dataset_name, subset_name)
    dataset = dataset[split_mapping[split]]
    dataset = utils.apply_template(dataset, template)
    return utils.hf_dataset_to_tf_dataset(dataset)


def add_task(dataset_name, subset_name, template_name, task_name=None, split_mapping=None):

    template = all_templates.get_dataset(dataset_name, subset_name)[template_name]

    task_name = task_name or utils.get_task_name(dataset_name, subset_name, template_name)
    # if task_name in CLEAN_EVAL_TASKS:
    #     metrics = EVAL_METRICS[task_name]
    # else:
    # TODO
    metrics = [t5.evaluation.metrics.sequence_accuracy]

    dataset_splits = utils.get_dataset_splits(dataset_name, subset_name)
    split_mapping = split_mapping or {k: k for k in dataset_splits.keys()}

    dataset_fn = functools.partial(
        get_tf_dataset,
        seed=None,
        dataset_name=dataset_name,
        subset_name=subset_name,
        template=template,
        split_mapping=split_mapping,
    )
    data_source = seqio.FunctionDataSource(
        dataset_fn,
        splits=list(split_mapping.keys()),
        num_input_examples={s: dataset_splits[split_mapping[s]].num_examples for s in split_mapping.keys()},
    )
    output_features = {
        "inputs": seqio.Feature(t5.data.get_default_vocabulary(), add_eos=False, dtype=tf.int32),
        "targets": seqio.Feature(t5.data.get_default_vocabulary(), add_eos=True, dtype=tf.int32),
    }
    preprocessors = [
        seqio.preprocessors.tokenize,
        seqio.preprocessors.append_eos,
        # seqio.CacheDatasetPlaceholder(required=False),  # TODO
    ]

    # Add train and normal eval tasks
    seqio.TaskRegistry.add(
        task_name,
        data_source,
        preprocessors=preprocessors,
        output_features=output_features,
        metric_fns=metrics,
        postprocess_fn=maybe_get_class_id_postprocessor(template),
    )

    # Add rank classification eval task
    labels = get_label_strings(template)
    if labels:
        rank_classification_preprocessor = functools.partial(
            t5.data.preprocessors.rank_classification,
            inputs_fn=lambda ex: tf.fill((len(labels),), ex["inputs"]),
            targets_fn=lambda ex: labels,
            is_correct_fn=lambda ex: tf.equal(labels, tf.strings.strip(ex["targets"])),
            weight_fn=lambda ex: 1.0,
        )
        seqio.TaskRegistry.add(
            task_name + "_score_eval",
            data_source,
            preprocessors=[rank_classification_preprocessor] + preprocessors,
            output_features=output_features,
            metric_fns=[functools.partial(t5.evaluation.metrics.rank_classification, num_classes=len(labels))],
            postprocess_fn=t5.data.postprocessors.rank_classification,
        )


train_sets = []
eval_sets = []
train_set_names = []
eval_set_names = []
experiment_path = pkg_resources.resource_filename(__name__, "experiment_D4.csv")
with open(experiment_path) as exp_file:
    reader = csv.DictReader(exp_file)
    for row in reader:
        if row['skip']:
            continue
        if row['subset'] == '':
            row['subset'] = None  # to match promptsource.Template object
        if row['do_train'] == 'TRUE':
            train_sets.append(row)
            train_set_names.append((row['HF_name'], row['subset']))
        if row['do_eval'] == 'TRUE':
            eval_sets.append(row)
            eval_set_names.append((row['HF_name'], row['subset']))

D4_names = train_set_names + eval_set_names
print(f'Number of training sets = {len(train_sets)}')
print(f'Number of evaluation sets = {len(eval_sets)}')

all_templates = promptsource.templates.TemplateCollection()
for dataset_name, subset_name in all_templates.keys:
    if (dataset_name, subset_name) not in train_set_names:  # D4_names:
        all_templates.remove(dataset_name, subset_name)
        continue
    # dataset = all_templates.get_dataset(dataset_name, subset)
    # for template_name in dataset.all_template_names:
    #     template = dataset[template_name]
        # if dataset_name == 'ropes':
        #     inspect(template.metadata)



for dataset_name, subset_name in all_templates.keys:

    if (dataset_name, subset_name) in DATASET_BLACKLIST:
        continue

    for template_name in all_templates.get_dataset(dataset_name, subset_name).all_template_names:
        add_task(dataset_name, subset_name, template_name)

print('Done here')


# Special case for ANLI, which has weirdly-named splits and rounds that should be subsets
dataset_name, subset_name = ("anli", None)
for anli_round in ("r1", "r2", "r3"):
    for template_name in all_templates.get_dataset(dataset_name, subset_name).all_template_names:
        task_name = utils.get_task_name(dataset_name, subset_name, template_name) + f"_{anli_round}"
        split_mapping = {
            "train": f"train_{anli_round}",
            "validation": f"dev_{anli_round}",
            "test": f"test_{anli_round}",
        }
        add_task(dataset_name, subset_name, template_name, task_name, split_mapping)


TASK_BLACKLIST = [
    # Tasks which often tokenize to > 1024 tokens currently
    "hotpot_qa_distractor_Generate_Explanations",
    "hotpot_qa_fullwiki_Generate_Explanations",
    "hotpot_qa_distractor_Generate_Answer_and_Explanations",
    "hotpot_qa_fullwiki_Generate_Answer_and_Explanations",
    "hotpot_qa_fullwiki_Generate_Answer",
    "hotpot_qa_distractor_Generate_Answer",
    "hotpot_qa_distractor_Generate_Title_2",
    "hotpot_qa_fullwiki_Generate_Title_2",
    "hotpot_qa_fullwiki_Generate_Title_1",
    "hotpot_qa_distractor_Generate_Title_1",
    "hotpot_qa_distractor_Generate_Question",
    "hotpot_qa_fullwiki_Generate_Question",
    "tab_fact_tab_fact_tab_fact_3",
    "tab_fact_tab_fact_tab_fact_2",
    "tab_fact_tab_fact_tab_fact_1",
    "tab_fact_tab_fact_tab_fact_7",
    "tab_fact_tab_fact_tab_fact_4",
    "tab_fact_tab_fact_tab_fact_5",
    "tab_fact_tab_fact_tab_fact_6",
    "wiki_hop_masked_Choose_Best_Object_Candidate",
    "wiki_hop_masked_Indirect_Question_about_Birthplace_Citizenship_Place_of_Death",
    "narrativeqa_Template_05",
    "ecthr_cases_alleged_violation_prediction_silver_rationales",
    # Tasks with broken cached files
    "gigaword_summarize_",
]

seqio.MixtureRegistry.add(
    "all_tasks_combined_max_1m",
    [task for task in seqio.TaskRegistry.names() if task not in TASK_BLACKLIST],
    default_rate=functools.partial(seqio.mixing_rate_num_examples, maximum=500_000),
)

seqio.MixtureRegistry.add(
    "all_super_glue_tasks",
    [task for task in seqio.TaskRegistry.names() if task.startswith("super_glue")],
    default_rate=seqio.mixing_rate_num_examples,
)


seqio.MixtureRegistry.add(
    "clean_tasks",
    [task for task in CLEAN_TASKS if task not in TASK_BLACKLIST],
    default_rate=functools.partial(seqio.mixing_rate_num_examples, maximum=500_000),
)


seqio.MixtureRegistry.add(
    "clean_eval_tasks",
    [task for task in CLEAN_EVAL_TASKS if task not in TASK_BLACKLIST],
    default_rate=functools.partial(seqio.mixing_rate_num_examples, maximum=500_000),
)

seqio.MixtureRegistry.add(
    "anli_eval_tasks",
    [task for task in CLEAN_EVAL_TASKS if task.startswith("anli")],
    default_rate=functools.partial(seqio.mixing_rate_num_examples, maximum=500_000),
)

seqio.MixtureRegistry.add(
    "score_eval_tasks",
    [task for task in seqio.TaskRegistry.names() if task.endswith("_score_eval")],
    default_rate=functools.partial(seqio.mixing_rate_num_examples, maximum=500_000),
)

seqio.MixtureRegistry.add(
    "clean_score_eval_tasks",
    [
        task
        for task in seqio.TaskRegistry.names()
        if task.endswith("_score_eval")
        and task.split("_score_eval")[0] in CLEAN_EVAL_TASKS
        and task.split("_score_eval")[0] not in TASK_BLACKLIST
    ],
    default_rate=functools.partial(seqio.mixing_rate_num_examples, maximum=500_000),
)
