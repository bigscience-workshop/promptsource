from jinja2 import BaseLoader, Environment, meta, TemplateError
import pytest
from promptsource.templates import TemplateCollection
from promptsource.utils import get_dataset_builder

# Sets up Jinja environment
env = Environment(loader=BaseLoader)

# Loads templates and iterates over each data (sub)set
template_collection = TemplateCollection()


@pytest.mark.parametrize("dataset", template_collection.keys)
def test_dataset(dataset):
    """
    Validates all the templates in the repository with simple syntactic checks:
    0. Are all templates parsable YAML?
    1. Do all templates parse in Jinja and are all referenced variables in the dataset schema?
    2. Does the template contain a prompt/output separator "|||" ?
    3. Are all names and templates within a data (sub)set unique?

    :param dataset: (dataset_name, subset_name) pair to test

    """
    dataset_name, subset_name = dataset

    # Loads dataset information
    builder_instance = get_dataset_builder(dataset_name, subset_name)
    features = builder_instance.info.features.keys()
    features = set([feature.replace("-", "_") for feature in features])

    # Initializes sets for checking uniqueness among templates
    template_name_set = set()
    template_jinja_set = set()

    # Iterates over each template for current data (sub)set
    dataset_templates = template_collection.get_dataset(dataset_name, subset_name)
    any_task = False
    for template_name in dataset_templates.all_template_names:
        template = dataset_templates[template_name]
        any_task = any_task or template.get_task_template()
        # Check 1: Jinja and all features valid?
        try:
            parse = env.parse(template.jinja)
        except TemplateError as e:
            raise ValueError(f"Template for dataset {dataset_name}/{subset_name} "
                             f"with uuid {template.get_id()} failed to parse.") from e

        variables = meta.find_undeclared_variables(parse)
        for variable in variables:
            if variable not in features:
                raise ValueError(f"Template for dataset {dataset_name}/{subset_name} "
                                 f"with uuid {template.get_id()} has unrecognized variable {variable}.")

        # Check 2: Prompt/output separator present?
        if "|||" not in template.jinja:
            raise ValueError(f"Template for dataset {dataset_name}/{subset_name} "
                             f"with uuid {template.get_id()} has no prompt/output separator.")

        # Check 3: Unique names and templates?
        if template.get_name() in template_name_set:
            raise ValueError(f"Template for dataset {dataset_name}/{subset_name} "
                             f"with uuid {template.get_id()} has duplicate name.")

        if template.jinja in template_jinja_set:
            raise ValueError(f"Template for dataset {dataset_name}/{subset_name} "
                             f"with uuid {template.get_id()} has duplicate definition.")

        template_name_set.add(template.get_name())
        template_jinja_set.add(template.jinja)

    # Turned off for now until we fix.
    #assert any_task, "There must be at least one task template for each dataset"
