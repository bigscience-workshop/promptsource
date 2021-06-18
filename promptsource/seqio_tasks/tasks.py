import datasets
import tensorflow as tf
import promptsource.templates
import seqio
import t5
import functools
from . import utils


all_templates = promptsource.templates.TemplateCollection()

for dataset_name, subset_name in all_templates.keys:

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

        seqio.TaskRegistry.add(
            utils.get_task_name(dataset_name, subset_name, template_name),
            seqio.FunctionDataSource(
                functools.partial(
                    dataset_fn,
                    seed=None,
                    dataset_name=dataset_name,
                    subset_name=subset_name,
                    template=template,
                ),
                splits=list(dataset_splits.keys()),
                num_input_examples={
                    s: dataset_splits[s].num_examples for s in dataset_splits.keys()
                }
            ),
            preprocessors=[
                seqio.preprocessors.tokenize, seqio.preprocessors.append_eos
            ],
            output_features={
                "inputs": seqio.Feature(
                    seqio.SentencePieceVocabulary(t5.data.DEFAULT_SPM_PATH),
                    add_eos=False, dtype=tf.int32
                ),
                "targets": seqio.Feature(
                    seqio.SentencePieceVocabulary(t5.data.DEFAULT_SPM_PATH),
                    add_eos=True, dtype=tf.int32
                ),
            },
            metric_fns=[t5.evaluation.metrics.sequence_accuracy]
        )
