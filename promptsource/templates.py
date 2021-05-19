import yaml
from jinja2 import Environment, BaseLoader, PackageLoader, select_autoescape
env = Environment(loader=BaseLoader)


def get_sample_template_data():
    data = TemplateCollection()

    ag_news_template = Template(
        'basic',
        'Example template.',
        'return example["text"] + "\n\nIs this an example of news about world politics, sports, business, or technology?"',
        'label_map = {\n'
        '    0: "World politics",\n'
        '    1: "Sports",\n'
        '    2: "Business",\n'
        '    3: "Technology"}\n'
        'return label_map[example["label"]]',
        )

    data.add_template("ag_news", ag_news_template)

    trec_template = Template(
        'basic',
        'Example template.',
        'return example["text"] + "\n\nIs this asking about a description, an entity, '
        'an abbreviation, a person, or a quantity?"',
        'label_map = {\n'
        '    0: "A description",\n'
        '    1: "An entity",\n'
        '    2: "An abbreviation",\n'
        '    3: "A person",\n'
        '    4: "A quantity"}\n'
        'return label_map[example["label-coarse"]]',
        )

    data.add_template("trec", trec_template)

    return data


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
            raise ValueError(f"No template with name {template_name} " +
                             f"for dataset {dataset} exists.")

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

    def __len__(self):
        size = 0
        for dataset in self.templates.keys():
            size += len(self.templates[dataset])
        return size


class Template(yaml.YAMLObject):
    """
    A prompt template.
    """
    yaml_tag = u'!Template'

    def __init__(self, name, reference, prompt_fn=None, output_fn=None, jinja_tpl=None):
        """
        Creates a prompt template.

        A prompt template is made up three main pieces: strings that define
        three functions, one each for generating the input, the prompt, and the
        output given an example. These strings should not include the function
        signature, but should assume that there is an input called "example".
        Each function should return a string.

        :param name: unique name (per dataset) for template
        :param reference: string metadata describing author or paper reference
                          for template
        :param prompt_fn: string defining function that creates prompt from example
        :param output_fn: string defining function that creates output from example
        """
        self.name = name
        self.prompt_fn = prompt_fn
        self.output_fn = output_fn
        self.reference = reference
        self.jinja = jinja_tpl

    @property
    def jinja_tpl(self):
        if hasattr(self, "jinja"):
            return self.jinja
        else:
            return ""
        
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
        :return: tuple of 3 strings, for input, prompt, and output
        """
        if self.jinja_tpl:
            rtemplate = env.from_string(self.jinja_tpl)
            return rtemplate.render(**example).split("|||")

        else:         
            fns = {}
            exec(self._make_fun_str("prompt_fn", ["example"], self.prompt_fn), fns)
            exec(self._make_fun_str("output_fn", ["example"], self.output_fn), fns)
            return (fns['prompt_fn'](example), fns['output_fn'](example))

    @classmethod
    def _make_fun_str(cls, name, args, body):
        """
        Creates a string representation of a Python function.

        :param name: the name of the function
        :param args: iterable of strings naming function arguments
        :param body: the function definition. The outermost context should be unindented.
        :return: full function definition that can be parsed by exec
        """
        arg_str = ", ".join(args)
        signature = f"def {name}({arg_str}):\n"
        body = "\n".join([("    " + line) for line in body.split("\n")])
        return signature + body


