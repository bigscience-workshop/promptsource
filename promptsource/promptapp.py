import pandas as pd
import streamlit as st
from utils import get_dataset, get_dataset_confs, list_datasets, removeHyphen, renameDatasetColumn, render_features

from templates import TemplateCollection


#
# Helper functions for datasets library
#

get_dataset = st.cache(allow_output_mutation=True)(get_dataset)
get_dataset_confs = st.cache(get_dataset_confs)

#
# Initial page setup
#
st.set_page_config(layout="wide")
mode = st.sidebar.selectbox(
    label="Choose a mode",
    options=["Helicopter view", "Prompted dataset viewer"],
    index=0,
    key="mode_select",
)
st.sidebar.title(f"Prompt sourcing 🌸 - {mode}")

#
# Loads template data
#
try:
    template_collection = TemplateCollection()
except FileNotFoundError:
    st.error(
        "Unable to find the templates folder!\n\n"
        "We expect the folder to be in the working directory. "
        "You might need to restart the app in the root directory of the repo."
    )
    st.stop()


if mode == "Helicopter view":
    st.title("High level metrics")
    st.write("We can improve these metrics, please contribute!")
    st.write(
        "If you want to take ownership for prompting a particular dataset, "
        + "put your name in [this spreadsheet](https://docs.google.com/spreadsheets/d/10SBt96nXutB49H52PV2Lvne7F1NvVr_WZLXD8_Z0JMw/edit?usp=sharing)."
    )

    counts = template_collection.get_templates_count()
    nb_prompted_datasets = len(counts)
    st.write(f"## Number of *prompted datasets*: `{nb_prompted_datasets}`")
    nb_prompts = sum(counts.values())
    st.write(f"## Number of *prompts*: `{nb_prompts}`")
    st.markdown("***")

    st.write("Details per dataset")
    results = []
    for (dataset_name, subset_name) in template_collection.keys:
        dataset_templates = template_collection.get_dataset(dataset_name, subset_name)
        results.append(
            {
                "Dataset name": dataset_name,
                "Subset name": "" if subset_name is None else subset_name,
                "Number of templates": len(dataset_templates),
                "Template names": [t.name for t in dataset_templates.templates.values()],
                # TODO: template name is not very informative... refine that
            }
        )
    results_df = pd.DataFrame(results)
    results_df.sort_values(["Number of templates"], inplace=True)
    results_df.reset_index(drop=True, inplace=True)
    st.table(results_df)

elif mode == "Prompted dataset viewer":
    #
    # Loads dataset information
    #

    dataset_list = list_datasets(
        template_collection,
        _priority_filter=False,
        _priority_max_templates=None,
        _state=None,
    )

    #
    # Select a dataset
    #
    dataset_key = st.sidebar.selectbox(
        "Dataset",
        dataset_list,
        key="dataset_select",
        help="Select the dataset to visualize.",
    )

    #
    # If a particular dataset is selected, loads dataset and template information
    #
    if dataset_key is not None:

        #
        # Check for subconfigurations (i.e. subsets)
        #
        configs = get_dataset_confs(dataset_key)
        conf_option = None
        if len(configs) > 0:
            conf_option = st.sidebar.selectbox("Subset", configs, index=0, format_func=lambda a: a.name)

        dataset, failed = get_dataset(dataset_key, str(conf_option.name) if conf_option else None)
        if failed:
            if dataset.manual_download_instructions is not None:
                st.error(f"Dataset {dataset_key} requires manual download. Please skip for the moment.")
            else:
                st.error(f"Loading dataset {dataset_key} failed.\n{dataset}. Please skip for the moment.")

        splits = list(dataset.keys())
        index = 0
        if "train" in splits:
            index = splits.index("train")
        split = st.sidebar.selectbox("Split", splits, key="split_select", index=index)
        dataset = dataset[split]
        dataset = renameDatasetColumn(dataset)

        dataset_templates = template_collection.get_dataset(dataset_key, conf_option.name if conf_option else None)

        template_list = dataset_templates.all_template_names
        num_templates = len(template_list)
        st.sidebar.write(
            "No of Templates created for "
            + f"`{dataset_key + (('/' + conf_option.name) if conf_option else '')}`"
            + f": **{str(num_templates)}**"
        )

        if num_templates > 0:
            template_name = st.sidebar.selectbox(
                "Template name",
                template_list,
                key="template_select",
                index=0,
                help="Select the template to visualize.",
            )

        step = 50
        example_index = st.sidebar.number_input(
            f"Select the example index (Size = {len(dataset)})",
            min_value=0,
            max_value=len(dataset) - step,
            value=0,
            step=step,
            key="example_index_number_input",
            help="Offset = 50.",
        )

        st.sidebar.subheader("Dataset Schema")
        st.sidebar.write(render_features(dataset.features))

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

        #
        # Display template information
        #
        if num_templates > 0:
            template = dataset_templates[template_name]
            st.subheader("Template")
            st.markdown("##### Name")
            st.text(template.name)
            st.markdown("##### Reference")
            st.text(template.reference)
            st.markdown("##### Jinja")
            st.text(template.jinja)
            st.markdown("***")

        #
        # Display a couple (steps) examples
        #
        for ex_idx in range(example_index, example_index + step):
            if ex_idx >= len(dataset):
                continue
            example = dataset[ex_idx]
            example = removeHyphen(example)
            col1, _, col2 = st.beta_columns([12, 1, 12])
            with col1:
                st.write(example)
            if num_templates > 0:
                with col2:
                    prompt = template.apply(example)
                    st.write("Prompt + X")
                    st.text(prompt[0])
                    if len(prompt) > 1:
                        st.write("Y")
                        st.text(prompt[1])
            st.markdown("***")
