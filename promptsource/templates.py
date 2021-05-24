import yaml
from jinja2 import BaseLoader, Environment
import itertools

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

    def add_template(self, dataset, num_templates, template):
        """
        Adds a new template for the dataset

        :param dataset: dataset key for the template
        :param template: template
        """
        if dataset not in self.templates:
            self.templates[dataset] = {}        
        self.templates[dataset][template.get_name()] = template
        self.templates[dataset]["count"] = num_templates

    def remove_template(self, dataset, num_templates, template_name):
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
        self.templates[dataset]["count"] = num_templates
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

        dataset_templates_copy = self.templates[dataset].copy()
        del dataset_templates_copy["count"]
        return dataset_templates_copy
    
    def get_templates_count(self):
        count_dict = {}
        for k,v in self.templates.items():
            if isinstance(k, str):
                count_dict[k] = v["count"]
        temp_with_conf = {k:v for k,v in self.templates.items() if isinstance(k, tuple)}
        groups = itertools.groupby(sorted(temp_with_conf), lambda x:(x[0]))
        for dataset, group in groups:
            count_dict[dataset] = sum(self.templates[conf]["count"] for conf in group)
        return count_dict


    def __len__(self):
        size = 0
        for dataset in self.templates.keys():
            size += len(self.templates[dataset])
        return size


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
