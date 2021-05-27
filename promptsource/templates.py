import itertools

import yaml
from jinja2 import BaseLoader, Environment
import os
from typing import Dict, List, Optional
import glob

TEMPLATES_FOLDER_PATH = './templates/'

env = Environment(loader=BaseLoader)

class TemplateCollection:
    """
    Collection of prompt templates.

    Templates are organized by dataset. The key for a dataset is either the
    dataset name or, f the dataset requires a configuration, the key is a
    tuple of (dataset name, configuration name).
    """

    def __init__(self):
        """
        Initializes an empty template collection.
        """
        # Templates are organized as a dictionary of dictionaries.
        # Outer dictionary is keyed by dataset key.
        # Inner dictionary is keyed by template name.
        self.templates = {}

    @classmethod
    def read_from_file(cls, file):
        """
        Reads a file containing a prompt collection.

        :param file: file-like object producing strings
        """
        template_dict = yaml.load(file, Loader=yaml.FullLoader)
        templates = TemplateCollection()
        templates.templates = template_dict
        return templates

    def write_to_file(self, file):
        """
        Writes to a file with the current prompt collection.

        :param file: file-like object supporting string inputs
        """
        yaml.dump(self.templates, file)

    def add_template(self, dataset, template):
        """
        Adds a new template for the dataset

        :param dataset: dataset key for the template
        :param template: template
        """
        if dataset not in self.templates:
            self.templates[dataset] = {}
        self.templates[dataset][template.get_name()] = template

    def remove_template(self, dataset, template_name):
        """
        Deletes a template

        :param dataset: dataset key for the template
        :param template_name: name of template to remove
        """
        if dataset not in self.templates:
            raise ValueError("No templates for dataset exist.")

        if template_name not in self.templates[dataset]:
            raise ValueError(f"No template with name {template_name} " + f"for dataset {dataset} exists.")

        del self.templates[dataset][template_name]
        if len(self.templates[dataset]) == 0:
            del self.templates[dataset]

    def get_templates(self, dataset):
        """
        Returns all templates for a dataset

        :param dataset: dataset key
        :return: copy of internal dictionary with template names as keys
        """
        if dataset not in self.templates:
            return {}
        return self.templates[dataset].copy()

    def get_templates_count(self):
        count_dict = {}
        for k, v in self.templates.items():
            if isinstance(k, str):
                count_dict[k] = len(v)
        temp_with_conf = {k: v for k, v in self.templates.items() if isinstance(k, tuple)}
        groups = itertools.groupby(sorted(temp_with_conf), lambda x: (x[0]))
        for dataset, group in groups:
            count_dict[dataset] = sum(len(self.templates[conf]) for conf in group)
        return count_dict

    def __len__(self):
        size = 0
        for dataset in self.templates.keys():
            size += len(self.templates[dataset])
        return size

class DatasetTemplates:
    """
    Collection of prompt templates.

    Templates are organized by dataset. The key for a dataset is either the
    dataset name or, f the dataset requires a configuration, the key is a
    tuple of (dataset name, configuration name).
    """

    TEMPLATES_KEY = 'templates'
    DATASET_KEY = 'dataset'
    CONFIG_KEY = 'config'

    def __init__(self, dataset_name: str, config_name: str = None):
        """
        Initializes an empty template collection.
        """
        # dictionary is keyed by template name.
        self.dataset_name = dataset_name
        self.config_name = config_name
        self.path = os.path.join(TEMPLATES_FOLDER_PATH, self.dataset_name)

        self.templates = self.read_from_file()

    @property
    def keys(self):
        return list(self.templates.keys())

    @property
    def folder_path(self):
        if self.config_name:
            return os.path.join(TEMPLATES_FOLDER_PATH, self.dataset_name, self.config_name)
        else:
            return os.path.join(TEMPLATES_FOLDER_PATH, self.dataset_name)

    @property
    def yaml_path(self):
        return os.path.join(self.folder_path, 'templates.yaml')

    def parse_yaml(self, yaml_dict):
        return yaml_dict['templates']

    def format_yaml(self):
        formatted_dict = {self.DATASET_KEY: self.dataset_name, self.TEMPLATES_KEY: self.templates}
        if self.config_name:
            formatted_dict[self.CONFIG_KEY] = self.config_name
        return formatted_dict

    def read_from_file(self) -> Dict:
        """
        Reads a file containing a prompt collection.

        :param file: file-like object producing strings
        """

        if not os.path.exists(self.yaml_path):
            return {}
        yaml_dict =  yaml.load(open(self.yaml_path, 'r'), Loader=yaml.FullLoader)
        return yaml_dict[self.TEMPLATES_KEY]

    def write_to_file(self):
        """
        Writes to a file with the current prompt collection.

        :param file: file-like object supporting string inputs
        """

        # We only create the folder if a template is written
        if not os.path.exists(self.folder_path):
            os.mkdir(self.folder_path)
        yaml.dump(self.format_yaml(), open(self.yaml_path, 'w'))

    def add_template(self, template):
        """
        Adds a new template for the dataset

        :param dataset: dataset key for the template
        :param template: template
        """
        self.templates[template.get_name()] = template

        self.write_to_file()

    def remove_template(self, template_name):
        """
        Deletes a template

        :param dataset: dataset key for the template
        :param template_name: name of template to remove
        """

        if template_name not in self.templates.keys():
            raise ValueError(f"No template with name {template_name} " + f"for dataset {self.dataset_name} exists.")

        del self.templates[template_name]

        self.write_to_file()

    def update_template(self, template, jinja, reference):
        self.templates[template.name].jinja = jinja
        self.templates[template.name].reference = reference

        self.write_to_file()

    def get_templates(self, dataset):
        """
        Returns all templates for a dataset

        :param dataset: dataset key
        :return: copy of internal dictionary with template names as keys
        """
        return self.templates.copy()

    def __getitem__(self, item):
        return self.templates[item]

    def __len__(self) -> int:
        return len(self.templates)


class NewTemplateCollection:
    def __init__(self):
        self.path = TEMPLATES_FOLDER_PATH
        self.datasets_templates: Dict[(str, Optional[str]), DatasetTemplates] = self._collect_dataset()

    @property
    def keys(self):
        return list(self.datasets_templates.keys())

    @staticmethod
    def parse_yaml_name(name):
        return name.replace('templates_', '').replace('.yaml', '')

    def _collect_dataset(self):
        dataset_folders = os.listdir(self.path)

        output = {} # format is {(dataset_name, config_name): DatasetsTemplates}
        for dataset in dataset_folders:
            for filename in os.listdir(os.path.join(self.path, dataset)):
                if filename.endswith('.yaml'):
                    output[(dataset, None)] = DatasetTemplates(dataset)
                else:
                    config = self.parse_yaml_name(filename)
                    output[(dataset, config)] = DatasetTemplates(dataset, config)
        return output

    def get_dataset(self, dataset_name: str, config_name: str = None) -> DatasetTemplates:
        if dataset_name not in self.keys:
            self.datasets_templates[(dataset_name, config_name)] = DatasetTemplates(dataset_name, config_name)

        return self.datasets_templates[(dataset_name, config_name)]

    def get_templates(self, dataset, config_name = None) -> DatasetTemplates:
        """
        Returns all templates for a dataset

        :param dataset: dataset key
        :return: copy of internal dictionary with template names as keys
        """
        if dataset not in self.datasets_templates:
            return {}
        return self.datasets_templates[(dataset, config_name)]

    def get_templates_count(self):

        from collections import defaultdict

        count_dict = defaultdict(int)
        for k, v in self.datasets_templates.items():
            # only taking dataset name
            count_dict[k[0]] += len(v)
        # converting to regular dict
        return dict(count_dict)


class Template(yaml.YAMLObject):
    """
    A prompt template.
    """

    yaml_tag = "!Template"

    def __init__(self, name, jinja, reference):
        """
        Creates a prompt template.

        A prompt template is expressed in Jinja. It is rendered using an example
        from the corresponding Hugging Face datasets library (a dictionary). The
        separator ||| should appear once to divide the template into prompt and
        output. Generally, the prompt should provide information on the desired
        behavior, e.g., text passage and instructions, and the output should be
        a desired response.

        :param name: unique name (per dataset) for template
        :param jinja: template expressed in Jinja
        :param reference: string metadata describing author or paper reference
                          for template
        """
        self.name = name
        self.jinja = jinja
        self.reference = reference

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

    def apply(self, example):
        """
        Creates a prompt by applying this template to an example

        :param example: the dataset example to create a prompt for
        :return: tuple of 2 strings, for prompt and output
        """
        rtemplate = env.from_string(self.jinja)
        return rtemplate.render(**example).split("|||")
