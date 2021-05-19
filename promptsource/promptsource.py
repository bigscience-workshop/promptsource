import datasets
import random
import streamlit as st

from session_state import get as get_session_state
from templates import Template, TemplateCollection

st.set_page_config(layout="wide")
st.title('PromptSource: Create Prompt Templates')

#
# Loads template data
#
try:
    with open("./templates.yaml", 'r') as f:
        templates = TemplateCollection.read_from_file(f)
except FileNotFoundError:
    st.error("Unable to load the templates file!\n\n"
             "We expect the file templates.yaml to be in the working directory. "
             "You might need to restart the app in the root directory of the repo.")
    st.stop()


def save_data(message="Done!"):
    with open("./templates.yaml", 'w') as f:
        templates.write_to_file(f)
        st.success(message)


#
# Loads dataset information
#
dataset_list = datasets.list_datasets()

#
# Initializes state
#
session_state = get_session_state(example_index=0, dataset=dataset_list[0])

#
# Select a dataset
#
# TODO: Currently raises an error if you select a dataset that requires a
# TODO: configuration. Not clear how to query for these options.
dataset_key = st.sidebar.selectbox('Dataset', dataset_list, key='dataset_select',
                 help='Select the dataset to work on. Number in parens ' +
                      'is the number of prompts created.')
st.sidebar.write("HINT: Try ag_news or trec for examples.")

#
# If a particular dataset is selected, loads dataset and template information
#
if dataset_key is not None:
    dataset = datasets.load_dataset(dataset_key)

    st.sidebar.subheader("Dataset Info")
    st.sidebar.markdown(f"[Hugging Face Page](https://huggingface.co/datasets/{dataset_key})")

    with st.form("example_form"):
        st.sidebar.subheader("Random Training Example")
        new_example_button = st.sidebar.button("New Example", key="new_example")
        split = st.sidebar.selectbox("Split", list(dataset.keys()))
        if split or new_example_button or dataset_key != session_state.dataset:
            session_state.example_index = random.randint(0, len(dataset[split]))
            session_state.dataset = dataset_key
        example = dataset[split][session_state.example_index]
        st.sidebar.write(example)

    col1, _, col2 = st.beta_columns([18, 1, 6])

    with col1:
        with st.beta_expander("Select Template", expanded=True):
            with st.form("new_template_form"):
                new_template_input = st.text_input("New Template Name", key="new_template_key", value="",
                                                   help="Enter name and hit enter to create a new template.")
                new_template_submitted = st.form_submit_button("Create")
                if new_template_submitted:
                    new_template_name = new_template_input
                    if new_template_name in templates.get_templates(dataset_key):
                        st.error(f"A template with the name {new_template_name} already exists "
                                 f"for dataset {dataset_key}.")
                    else:
                        template = Template(new_template_name, 'return ""', 'return ""', 'return ""', "")
                        templates.add_template(dataset_key, template)
                        save_data()
                else:
                    new_template_name = None

            dataset_templates = templates.get_templates(dataset_key)
            template_list = list(dataset_templates.keys())
            if new_template_name:
                index = template_list.index(new_template_name)
            else:
                index = 0
            template_name = st.selectbox('', template_list, key='template_select',
                                          index=index, help='Select the template to work on.')

            if st.button("Delete Template", key="delete_template"):
                templates.remove_template(dataset_key, template.get_name())
                save_data("Template deleted!")

        #
        # If template is selected, displays template editor
        #
        if template_name is not None:
            with st.form("edit_template_form"):
                template = dataset_templates[template_name]

                code_height = 40
                input_fn_code = st.text_area('Input Function', height=code_height, value=template.input_fn)
                prompt_fn_code = st.text_area('Prompt Function', height=code_height, value=template.prompt_fn)
                output_fn_code = st.text_area('Output Function', height=code_height, value=template.output_fn)

                reference = st.text_area("Template Reference",
                                         help="Your name and/or paper reference.",
                                         value=template.reference)

                if st.form_submit_button("Save"):
                    template.input_fn = input_fn_code
                    template.prompt_fn = prompt_fn_code
                    template.output_fn = output_fn_code
                    template.reference = reference
                    save_data()

    #
    # Displays template output on current example if a template is selected
    # (in second column)
    #
    with col2:
        if template_name is not None:
            st.empty()
            st.subheader("Template Output")
            template = dataset_templates[template_name]
            prompt = template.apply(example)
            st.write(prompt[0])
            st.write(prompt[1])
            st.write(prompt[2])
