import datasets
import requests
import streamlit as st
from session import _get_state
from templates import Template, TemplateCollection
from utils import _ADDITIONAL_ENGLISH_DATSETS

#
# Helper functions for datasets library
#


@st.cache(allow_output_mutation=True)
def get_dataset(path, conf=None):
    "Get a dataset from name and conf."
    module_path = datasets.load.prepare_module(path, dataset=True)
    builder_cls = datasets.load.import_main_class(module_path[0], dataset=True)
    if conf:
        builder_instance = builder_cls(name=conf, cache_dir=None)
    else:
        builder_instance = builder_cls(cache_dir=None)
    fail = False
    if builder_instance.manual_download_instructions is None and builder_instance.info.size_in_bytes is not None:
        builder_instance.download_and_prepare()
        dts = builder_instance.as_dataset()
        dataset = dts
    else:
        dataset = builder_instance
        fail = True
    return dataset, fail


@st.cache
def get_dataset_confs(path):
    "Get the list of confs for a dataset."
    module_path = datasets.load.prepare_module(path, dataset=True)
    # Get dataset builder class from the processing script
    builder_cls = datasets.load.import_main_class(module_path[0], dataset=True)
    # Instantiate the dataset builder
    confs = builder_cls.BUILDER_CONFIGS
    if confs and len(confs) > 1:
        return confs
    return []


def render_features(features):
    if isinstance(features, dict):
        return {k: render_features(v) for k, v in features.items()}
    if isinstance(features, datasets.features.ClassLabel):
        return features.names

    if isinstance(features, datasets.features.Value):
        return features.dtype

    if isinstance(features, datasets.features.Sequence):
        return {"[]": render_features(features.feature)}
    return features


def reset_template_state():
    state.template_name = None
    state.jinja = None
    state.reference = None


#
# Loads session state
#
state = _get_state()

#
# Initial page setup
#
st.set_page_config(layout="wide")
st.sidebar.title("PromptSource 🌸")

#
# Loads template data
#
try:
    with open("./templates.yaml", "r") as f:
        templates = TemplateCollection.read_from_file(f)
except FileNotFoundError:
    st.error(
        "Unable to load the templates file!\n\n"
        "We expect the file templates.yaml to be in the working directory. "
        "You might need to restart the app in the root directory of the repo."
    )
    st.stop()


def save_data(message="Done!"):
    with open("./templates.yaml", "w") as f:
        templates.write_to_file(f)
        st.success(message)


#
# Loads dataset information
#
def filter_english_datasets():
    """Filter English datasets based on language tags in metadata"""
    english_datasets = []

    response = requests.get("https://huggingface.co/api/datasets?full=true")
    tags = response.json()

    for dataset in tags:
        dataset_name = dataset["id"]

        is_community_dataset = "/" in dataset_name
        if is_community_dataset:
            continue

        if "card_data" not in dataset:
            continue
        metadata = dataset["card_data"]

        if "languages" not in metadata:
            continue
        languages = metadata["languages"]

        if "en" in languages:
            english_datasets.append(dataset_name)

    all_english_datasets = list(set(english_datasets + _ADDITIONAL_ENGLISH_DATSETS))
    return sorted(all_english_datasets)


def list_datasets(option):
    dataset_list = filter_english_datasets()
    count_dict = templates.get_templates_count()
    if option:
        dataset_list = list(
            set(dataset_list)
            - set(
                list(
                    d for d in count_dict if count_dict[d] > priority_max_templates and d != state.working_priority_ds
                )
            )
        )
        dataset_list.sort()
    return dataset_list


option = st.sidebar.checkbox("Filter Priority Datasets")
if option:
    priority_max_templates = st.sidebar.number_input(
        "Max no of templates per dataset", min_value=0, max_value=50, value=2, step=1
    )
else:
    # Clear working priority dataset retained in the
    # priority list with more than priority_max_templates
    state.working_priority_ds = None

dataset_list = list_datasets(option)

#
# Select a dataset
#
dataset_key = st.sidebar.selectbox(
    "Dataset",
    dataset_list,
    key="dataset_select",
    help="Select the dataset to work on. Number in parens " + "is the number of prompts created.",
)
st.sidebar.write("HINT: Try ag_news or trec for examples.")

# On dataset change, clear working priority dataset
# retained in the priority list with more than priority_max_templates
if dataset_key != state.working_priority_ds:
    state.working_priority_ds = None

#
# If a particular dataset is selected, loads dataset and template information
#
if dataset_key is not None:

    #
    # Check for subconfigurations
    #
    configs = get_dataset_confs(dataset_key)
    conf_avail = len(configs) > 0
    conf_option = None
    if conf_avail:
        start = 0
        conf_option = st.sidebar.selectbox("Subset", configs, index=start, format_func=lambda a: a.name)

    dataset, _ = get_dataset(dataset_key, str(conf_option.name) if conf_option else None)

    k = list(dataset.keys())
    index = 0
    if "train" in dataset.keys():
        index = k.index("train")
    split = st.sidebar.selectbox("Split", k, index=index)
    dataset = dataset[split]

    #
    # Display dataset information
    #
    st.header("Dataset: " + dataset_key + " " + (("/ " + conf_option.name) if conf_option else ""))

    st.markdown(
        "*Homepage*: "
        + dataset.info.homepage
        + "\n\n*Dataset*: https://github.com/huggingface/datasets/blob/master/datasets/%s/%s.py"
        % (dataset_key, dataset_key)
    )

    md = """
    %s
    """ % (
        dataset.info.description.replace("\\", "") if dataset_key else ""
    )
    st.markdown(md)

    st.sidebar.subheader("Dataset Schema")
    st.sidebar.write(render_features(dataset.features))

    template_key = dataset_key
    if conf_option:
        template_key = (dataset_key, conf_option.name)
    dataset_templates = templates.get_templates(template_key)
    template_list = list(dataset_templates.keys())
    num_templates = len(template_list)
    st.sidebar.subheader(
        "No of Templates created for: " + dataset_key + (("/ " + conf_option.name) if conf_option else "")
    )
    st.sidebar.write(num_templates)

    st.sidebar.subheader("Select Example")
    example_index = st.sidebar.slider("Select the example index", 0, len(dataset) - 1)

    example = dataset[example_index]
    st.sidebar.write(example)

    col1, _, col2 = st.beta_columns([18, 1, 6])

    # current_templates_key and state.templates_key are keys for the templates object
    current_templates_key = dataset_key
    if conf_option:
        current_templates_key = (dataset_key, conf_option.name)

    # Resets state if there has been a change in templates_key
    if state.templates_key != current_templates_key:
        state.templates_key = current_templates_key
        reset_template_state()

    with col1:
        with st.beta_expander("Select Template", expanded=True):
            with st.form("new_template_form"):
                new_template_name = st.text_input(
                    "New Template Name",
                    key="new_template_key",
                    value="",
                    help="Enter name and hit enter to create a new template.",
                )
                new_template_submitted = st.form_submit_button("Create")
                if new_template_submitted:
                    if new_template_name in templates.get_templates(state.templates_key):
                        st.error(
                            f"A template with the name {state.new_template_name} already exists "
                            f"for dataset {state.templates_key}."
                        )
                    elif new_template_name == "":
                        st.error("Need to provide a template name.")
                    else:
                        template = Template(new_template_name, "", "")
                        templates.add_template(state.templates_key, template)
                        save_data()
                        reset_template_state()
                        state.template_name = new_template_name
                        # Keep the current working dataset in priority list
                        if option:
                            state.working_priority_ds = dataset_key
                else:
                    state.new_template_name = None

            dataset_templates = templates.get_templates(state.templates_key)
            template_list = list(dataset_templates.keys())
            if state.template_name:
                index = template_list.index(state.template_name)
            else:
                index = 0
            state.template_name = st.selectbox(
                "", template_list, key="template_select", index=index, help="Select the template to work on."
            )

            if st.button("Delete Template", key="delete_template"):
                templates.remove_template(state.templates_key, state.template_name)
                save_data("Template deleted!")
                reset_template_state()

        if state.template_name is not None:
            template = dataset_templates[state.template_name]
            #
            # If template is selected, displays template editor
            #
            with st.form("edit_template_form"):
                state.jinja = st.text_area("Template", height=40, value=template.jinja)

                state.reference = st.text_area(
                    "Template Reference", help="Your name and/or paper reference.", value=template.reference
                )

                if st.form_submit_button("Save"):
                    template.jinja = state.jinja
                    template.reference = state.reference
                    save_data()
    #
    # Displays template output on current example if a template is selected
    # (in second column)
    #
    with col2:
        if state.template_name is not None:
            st.empty()
            st.subheader("Template Output")
            template = dataset_templates[state.template_name]
            prompt = template.apply(example)
            st.write(prompt[0])
            if len(prompt) > 1:
                st.write(prompt[1])

#
# Must sync state at end
#
state.sync()
