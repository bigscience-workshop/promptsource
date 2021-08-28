import os
import random
import uuid
from collections import Counter, defaultdict
from shutil import rmtree
from typing import Dict, List, Optional, Tuple

import pkg_resources
import yaml
from jinja2 import BaseLoader, Environment


# Truncation of jinja template variables
# 1710 = 300 words x 4.7 avg characters per word + 300 spaces
TEXT_VAR_LENGTH = 2048

# Local path to the folder containing the templates
TEMPLATES_FOLDER_PATH = pkg_resources.resource_filename(__name__, "templates")

env = Environment(loader=BaseLoader)

# Allow the python function zip()
env.globals.update(zip=zip)


def highlight(input):
    return "<span style='color: #F08080'>" + input + "</span>"


def choice(choices):
    return random.choice(choices)


def most_frequent(items):
    """Returns the set of items which appear most frequently in the input"""
    if not items:
        return
    item_counts = Counter(items).most_common()
    max_freq = item_counts[0][1]
    most_frequent_items = [c[0] for c in item_counts if c[1] == max_freq]
    return most_frequent_items


env.filters["highlight"] = highlight
env.filters["choice"] = choice
env.filters["most_frequent"] = most_frequent


class TemplateCollection:
    """
    This helper class wraps the DatasetTemplates class
    - Initialized the DatasetTemplates for all existing template folder
    - Give access to each DatasetTemplates
    - Provides aggregated counts over all DatasetTemplates
    """

    def __init__(self):

        # Dict of all the DatasetTemplates, key is the tuple (dataset_name, subset_name)
        self.datasets_templates: Dict[(str, Optional[str]), DatasetTemplates] = self._collect_dataset()

    @property
    def keys(self):
        return list(self.datasets_templates.keys())

    def _collect_dataset(self) -> Dict[Tuple[str, str], "DatasetTemplates"]:
        """
        Initialize a DatasetTemplates object for each templates.yaml detected in the templates folder

        Returns: a dict with key=(dataset_name, subset_name)
        """
        dataset_folders = os.listdir(TEMPLATES_FOLDER_PATH)
        dataset_folders = [folder for folder in dataset_folders if not folder.startswith(".")]

        output = {}  # format is {(dataset_name, subset_name): DatasetsTemplates}
        for dataset in dataset_folders:
            for filename in os.listdir(os.path.join(TEMPLATES_FOLDER_PATH, dataset)):
                if filename.endswith(".yaml"):
                    # If there is no sub-folder, there is no subset for this dataset
                    output[(dataset, None)] = DatasetTemplates(dataset)
                else:
                    # This is a subfolder, and its name corresponds to the subset name
                    output[(dataset, filename)] = DatasetTemplates(dataset_name=dataset, subset_name=filename)
        return output

    def get_dataset(self, dataset_name: str, subset_name: Optional[str] = None) -> "DatasetTemplates":
        """
        Return the DatasetTemplates object corresponding to the dataset name

        :param dataset_name: name of the dataset to get
        :param subset_name: name of the subset
        """
        # if the dataset does not exist, we add it
        if dataset_name not in self.keys:
            self.datasets_templates[(dataset_name, subset_name)] = DatasetTemplates(dataset_name, subset_name)

        return self.datasets_templates[(dataset_name, subset_name)]

    def get_templates_count(self) -> Dict:
        """
        Return the overall number count over all datasets

        NB: we don't breakdown datasets into subsets for the count, i.e subsets count are included
        into the dataset count
        """

        count_dict = defaultdict(int)
        for k, v in self.datasets_templates.items():
            # Subsets count towards dataset count
            count_dict[k[0]] += len(v)
        # converting to regular dict
        return dict(count_dict)


class DatasetTemplates:
    """
    Class that wraps all templates for a specific dataset/subset and implements all the helper
    functions necessary to read/write to the yaml file
    """

    TEMPLATES_KEY = "templates"
    DATASET_KEY = "dataset"
    SUBSET_KEY = "subset"
    TEMPLATE_FILENAME = "templates.yaml"

    def __init__(self, dataset_name: str, subset_name: str = None):
        self.dataset_name: str = dataset_name
        self.subset_name: str = subset_name
        # dictionary is keyed by template name.
        self.templates: Dict = self.read_from_file()

        # Mapping from template name to template id
        self.name_to_id_mapping = {}
        self.sync_mapping()

    def sync_mapping(self) -> None:
        """
        Re-compute the name_to_id_mapping to ensure it is in sync with self.templates
        """
        self.name_to_id_mapping = {template.name: template.id for template in self.templates.values()}

    @property
    def all_template_names(self) -> List[str]:
        """
        Sorted list of all templates names for this dataset
        """
        return sorted([template.name for template in self.templates.values()])

    @property
    def folder_path(self) -> str:
        if self.subset_name:
            return os.path.join(TEMPLATES_FOLDER_PATH, self.dataset_name, self.subset_name)
        else:
            return os.path.join(TEMPLATES_FOLDER_PATH, self.dataset_name)

    @property
    def yaml_path(self) -> str:
        return os.path.join(self.folder_path, self.TEMPLATE_FILENAME)

    def format_for_dump(self) -> Dict:
        """
        Create a formatted dictionary for the class attributes
        """
        formatted_dict = {self.DATASET_KEY: self.dataset_name, self.TEMPLATES_KEY: self.templates}
        if self.subset_name:
            formatted_dict[self.SUBSET_KEY] = self.subset_name
        return formatted_dict

    def read_from_file(self) -> Dict:
        """
        Reads a file containing a prompt collection.
        """

        if not os.path.exists(self.yaml_path):
            return {}
        yaml_dict = yaml.load(open(self.yaml_path, "r"), Loader=yaml.FullLoader)
        return yaml_dict[self.TEMPLATES_KEY]

    def write_to_file(self) -> None:
        """
        Writes to a file with the current prompt collection.
        """
        # Sync the mapping
        self.sync_mapping()

        # We only create the folder if a template is written
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)
        yaml.dump(self.format_for_dump(), open(self.yaml_path, "w"))

    def add_template(self, template: "Template") -> None:
        """
        Adds a new template for the dataset

        :param template: template
        """
        self.templates[template.get_id()] = template

        self.write_to_file()

    def remove_template(self, template_name: str) -> None:
        """
        Deletes a template

        :param template_name: name of template to remove
        """

        # Even if we have an ID, we want to check for duplicate names
        if template_name not in self.all_template_names:
            raise ValueError(f"No template with name {template_name} for dataset {self.dataset_name} exists.")

        del self.templates[self.name_to_id_mapping[template_name]]

        if len(self.templates) == 0:
            # There is no remaining template, we can remove the entire folder
            self.delete_folder()
        else:
            # We just update the file
            self.write_to_file()

    def update_template(
        self,
        current_template_name: str,
        new_template_name: str,
        jinja: str,
        reference: str,
        task_template: bool,
        answer_choices: List[str],
    ) -> None:
        """
        Updates a pre-existing template and writes changes

        :param current_template_name: current name of the template stored in self.templates
        :param new_template_name: new name for the template
        :param jinja: new jinja entry
        :param reference: new reference entry
        :param task_template: new task_template value
        :param answer_choices: new answer_choices list
        """
        template_id = self.name_to_id_mapping[current_template_name]
        self.templates[template_id].name = new_template_name
        self.templates[template_id].jinja = jinja
        self.templates[template_id].reference = reference
        self.templates[template_id].task_template = task_template
        self.templates[template_id].answer_choices = answer_choices

        self.write_to_file()

    def delete_folder(self) -> None:
        """
        Delete the folder corresponding to self.folder_path
        """
        self.sync_mapping()

        rmtree(self.folder_path)

        # If it is a subset, we have to check whether to remove the dataset folder
        if self.subset_name:
            # have to check for other folders
            base_dataset_folder = os.path.join(TEMPLATES_FOLDER_PATH, self.dataset_name)
            if len(os.listdir(base_dataset_folder)) == 0:
                rmtree(base_dataset_folder)

    def __getitem__(self, template_key: str) -> "Template":
        return self.templates[self.name_to_id_mapping[template_key]]

    def __len__(self) -> int:
        return len(self.templates)


class Template(yaml.YAMLObject):
    """
    A prompt template.
    """

    yaml_tag = "!Template"

    def __init__(self, name, jinja, reference, task_template=False, answer_choices=None):
        """
        Creates a prompt template.

        A prompt template is expressed in Jinja. It is rendered using an example
        from the corresponding Hugging Face datasets library (a dictionary). The
        separator ||| should appear once to divide the template into prompt and
        output. Generally, the prompt should provide information on the desired
        behavior, e.g., text passage and instructions, and the output should be
        a desired response.

        :param id: unique identifier to use as key in the yaml file
        :param name: unique name (per dataset) for template
        :param jinja: template expressed in Jinja
        :param reference: string metadata describing author or paper reference
                          for template
        :param task_template: bool whether this template corresponds 1-1 with the dataset task
        :param answer_choices: list of strings that enumerates the possible completions
                               for templates that should be evaluated as ranked
                               completions. If None, then the template is open-ended.
                               This list is accessible from within Jinja as the
                               variable `answer_choices`.

        """
        self.id = str(uuid.uuid4())
        self.name = name
        self.jinja = jinja
        self.reference = reference
        self.task_template = task_template
        self.answer_choices = answer_choices
        self.metadata = Template.Metadata()

    def get_id(self):
        """
        Returns the id of the template

        :return: unique id for template
        """
        return self.id

    def get_name(self):
        """
        Returns the name of the template

        :return: unique (per dataset) name for template
        """
        return self.name

    def get_reference(self):
        """
        Returns the bibliographic reference (or author) for the template

        :return: reference as a string
        """
        return self.reference

    def get_task_template(self):
        """
        Returns whether this template corresponds to the original or usual task
        for this dataset.

        task_template is just another piece of metadata stored in the
        Template.Metadata object. This method is for backwards compatibility.

        :return: bool
        """
        return self.metadata.task_template

    def get_answer_choices(self):
        """
        Returns a list of strings enumerating the possible completions for
        this template, or None if the template is open ended.

        :return: List[String]
        """
        return self.answer_choices

    def apply(self, example, truncate=True, highlight_variables=False):
        """
        Creates a prompt by applying this template to an example

        :param example: the dataset example to create a prompt for
        :param truncate: if True, example fields will be truncated to TEXT_VAR_LENGTH chars
        :param highlight_variables: highlight the added variables
        :return: tuple of 2 strings, for prompt and output
        """
        jinja = self.jinja

        # Truncates the prompt if needed
        if truncate:
            trunc_command = (
                f" | string | truncate({TEXT_VAR_LENGTH}) }}}}"  # Escaping curly braces requires doubling them
            )
            jinja = jinja.replace("}}", trunc_command)

        # Highlights text that was substituted for variables, if requested
        if highlight_variables:
            jinja = jinja.replace("}}", " | highlight }}")
        rtemplate = env.from_string(jinja)

        # Replaces any occurrences of the "|||" separator in the example, which
        # which will be replaced back after splitting
        pipe_protector = "3ed2dface8203c4c9dfb1a5dc58e41e0"
        protected_example = {
            key: value.replace("|||", pipe_protector) if isinstance(value, str) else value
            for key, value in example.items()
        }

        # Adds in answer_choices variable
        if "answer_choices" in protected_example:
            raise ValueError("Example contains the restricted key 'answer_choices'.")
        protected_example["answer_choices"] = self.answer_choices

        # Renders the Jinja template
        rendered_example = rtemplate.render(**protected_example)

        # Splits on the separator, and then replaces back any occurrences of the
        # separator in the original example
        return [part.replace(pipe_protector, "|||") for part in rendered_example.split("|||")]

    class Metadata(yaml.YAMLObject):
        """
        Metadata for a prompt template.
        """

        yaml_tag = "!TemplateMetadata"
        TRIVIAL_CHOICES = ('yes', 'no', 'true', 'false')

        def __init__(
                self,
                task_format: str,  # dropdown choice of 'classification', 'generation', 'extraction'
                original_task: bool,
                contributor: str,
                metric: Optional[str] = None,
                comment: Optional[str] = None,

                # if task_format == classification
                choices_in_prompt: Optional[bool] = None,
                fixed_choices: Optional[bool] = None,
                choices_fieldname: Optional[str] = None,

                # internal flags
                _do_train: Optional[bool] = None,
                _do_eval: Optional[bool] = None,
                _comment: Optional[str] = None,
                _version: Optional[str] = '2021/09',  # maybe change to datetime today
                ):
            """
            Initializes template metadata.

            :param original_task: If True, this prompt asks a model to perform the original task designed for
                this dataset.
            :param contributor: Full name of the person who adds this prompt to PromptSource. Note this is not
                the same as the original author of this prompt, which should be cited in the `comment` field
                (or the old `reference` field).
            :param metric: Each prompt could potentially use different metrics, especially for non-original
                task prompts.
            :param comment: Notes on how this prompt differs from others, who is the original author, what are
                some potential issues, etc.
            :param choices_in_prompt: If True, the answer choices are included in the templates such that models
                see those choices in the input. Only applicable to classification tasks.
            :param fixed_choices: If True, the answer choices are a fixed set of options (like "yes"/"no"/"maybe")
                that are the same for every example in the dataset. If False, each example has its own set of answer
                choices. Only applicable to classification tasks.
            :param choices_fieldname: The Hugging Face `datasets` fieldname for the example-specific answer choices,
                e.g., `entities` for SuperGLUE ReCoRD, `options` for RACE. Only applicable to classification tasks
                with fixed_choices == False.
            """
            assert task_format in ('classification', 'generation', 'extraction')
            if task_format == 'classification':
                assert fixed_choices is not None
                if fixed_choices:
                    self.trivial_choices = True
                    for choice in self.answer_choices:
                        if choice.lower() not in self.TRIVIAL_CHOICES:
                            self.trivial_choices = False
                    self.num_classes = len(self.answer_choices)
                else:  # example-specific answer choices
                    assert choices_fieldname is not None, 'Need to load per example choices for rank clasification.'

            self.task_format = task_format
            self.original_task = original_task
            self.contributor = contributor
            self.metric = metric
            self.comment = comment

            self.choices_in_prompt = choices_in_prompt
            self.fixed_choices = fixed_choices
            self.choices_fieldname = choices_fieldname

            self._do_train = _do_train
            self._do_eval = _do_eval
            self._comment = _comment
            self._version = _version


        def init_from_legacy_annotation(self, row: Dict[str, str]):
            """
            Initializes template metadata from Albert's CSV annotations.
            """
            self._version = '2021/06'

            if row['comment']:
                self._internal_comment = row['comment'] + '; '
            else:
                self._internal_comment = ''

            self._do_train = not bool(row['skip_train'])
            self._do_eval = bool(row['do_eval'])

            if row['nontrivial_choices_given']:
                self.task_format = 'classification'
                self.trivial_choices = False
                self.choices_in_prompt = True
                self.fixed_choices = 'unknown'  # HACK type error
                self.num_classes = 'unknown'
            elif row['nontrivial_choices_hidden']:
                self.task_format = 'classification'
                self.trivial_choices = False
                self.choices_in_prompt = False
                self.fixed_choices = 'unknown'
                self.num_classes = 'unknown'
            elif row['trivial_choices_given']:
                self.task_format = 'classification'
                self.trivial_choices = True
                self.choices_in_prompt = True
                self.fixed_choices = True
                self.num_classes = 2
            elif row['trivial_choices_hidden']:
                self.task_format = 'classification'
                self.trivial_choices = False
                self.choices_in_prompt = False
                self.fixed_choices = True
                self.num_classes = 2
            elif row['generative_non_true_task']:
                self.task_format = 'generation'
                self.original_task = False
            elif row['generative_non_true_implausible']:
                self.task_format = 'generation'
                self.original_task = False
                self._comment += 'implausibly open-ended;'
            elif row['generative_true_task']:
                self.task_format = 'generation'
                self.original_task = True
            elif row['answer_span_indices']:
                self.task_format = 'extraction'
            else:
                self.task_format = 'unknown'

            # # TODO probably not make flags of them, just append to _comment
            # if row['negated_answers']:
            #     self._comment += 'prompt asks for negated answers; '
            # self.negated_answers = negated_answers
            # self.counting = counting
            # self.long_distance = long_distance
            # self.no_sep_2_sentences = no_sep_2_sentences
            # self.answer_span_indices = answer_span_indices
            # self.non_natural_language = non_natural_language
