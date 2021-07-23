import functools

import datasets
import seqio
import t5
import tensorflow as tf

import promptsource.templates

from . import load_annotated_prompts, utils


# Tasks deemed as clean/useful
annotated_tasks = load_annotated_prompts.load_annotated_prompts()
CLEAN_TASKS = [t["dataset_subset_template"] for t in annotated_tasks if not t["skip_train"]]
CLEAN_EVAL_TASKS = [t["dataset_subset_template"] for t in annotated_tasks if t["do_eval"]]
EVAL_METRICS = {t["dataset_subset_template"]: t["metrics"] for t in annotated_tasks if t["do_eval"]}


# Datasets that don't work currently...
DATASET_BLACKLIST = [
    ("species_800", None),
    ("drop", None),
    ("discofuse", "discofuse-sport"),
    ("discofuse", "discofuse-wikipedia"),
    ("adversarial_qa", "adversarialQA"),
    ("tweet_eval", "emotion"),
    ("tweet_eval", "emoji"),
    ("tweet_eval", "hate"),
    ("tweet_eval", "offensive"),
    ("tweet_eval", "stance_atheism"),
    ("tweet_eval", "stance_abortion"),
    ("tweet_eval", "stance_feminist"),
    ("tweet_eval", "stance_climate"),
    ("tweet_eval", "sentiment"),
    ("tweet_eval", "stance_hillary"),
    ("tweet_eval", "irony"),
]


def strip_whitespace(output_or_target, example=None, is_target=False):
    """Cached tasks from promptsource all have a leading space on the ground-truth targets."""
    return output_or_target.strip()


all_templates = promptsource.templates.TemplateCollection()

for dataset_name, subset_name in all_templates.keys:

    if (dataset_name, subset_name) in DATASET_BLACKLIST:
        continue

    dataset_splits = utils.get_dataset_splits(dataset_name, subset_name)
    templates = all_templates.get_dataset(dataset_name, subset_name)

    for template_name in templates.all_template_names:

        template = templates[template_name]

        def dataset_fn(split, shuffle_files, seed, dataset_name, subset_name, template):
            # HF datasets does not support file-level shuffling
            del shuffle_files, seed
            dataset = datasets.load_dataset(dataset_name, subset_name)
            dataset = dataset[split]
            dataset = utils.apply_template(dataset, template)
            return utils.hf_dataset_to_tf_dataset(dataset)

        task_name = utils.get_task_name(dataset_name, subset_name, template_name)
        if task_name in CLEAN_EVAL_TASKS:
            metrics = EVAL_METRICS[task_name]
        else:
            metrics = [t5.evaluation.metrics.sequence_accuracy]

        seqio.TaskRegistry.add(
            task_name,
            seqio.FunctionDataSource(
                functools.partial(
                    dataset_fn,
                    seed=None,
                    dataset_name=dataset_name,
                    subset_name=subset_name,
                    template=template,
                ),
                splits=list(dataset_splits.keys()),
                num_input_examples={s: dataset_splits[s].num_examples for s in dataset_splits.keys()},
            ),
            preprocessors=[
                seqio.preprocessors.tokenize,
                seqio.preprocessors.append_eos,
                seqio.CacheDatasetPlaceholder(required=False),
            ],
            output_features={
                "inputs": seqio.Feature(t5.data.get_default_vocabulary(), add_eos=False, dtype=tf.int32),
                "targets": seqio.Feature(t5.data.get_default_vocabulary(), add_eos=True, dtype=tf.int32),
            },
            metric_fns=metrics,
            postprocess_fn=strip_whitespace,
        )

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
    default_rate=functools.partial(seqio.mixing_rate_num_examples, maximum=1000000),
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
