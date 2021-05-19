# PromptSource
Toolkit for collecting and applying templates of prompting instances.

WIP

## Setup
1. Download the repo
2. Navigate to root directory of the repo
3. Install requirements with `pip install -r requirements.txt`

## Running
From the root directory of the repo, you can launch the editor with
```
streamlit run promptsource/promptsource.py
```

## Writing Templates
Templates are currently represented as three python functions, one to generate each
piece of a prompting instance: input, prompt, and output.

You can write arbitrary python in the textboxes. Currently, each piece of code will be
inserted into a function with a header such as
```
def output_fn(example):
```
To write a function, fill in just the body of the function. Use no indentation for the
top-level context.

For example, to define an output function for ag_news, you could write
```python
label_map = {
    0: "World politics",
    1: "Sports",
    2: "Business",
    3: "Technology"}
return label_map[example["label"]]
```

## Contributing
This is very much a work in progress, and help is needed and appreciated. Anyone wishing to
contribute code can contact Steve Bach for commit access, or submit PRs from forks. Some particular
places you could start:
1. Try to express things! Explore a dataset and tell us what's hard to do to create templates you want
2. Look in the literature. Are there prompt creation methods that do/do not fit well right now?
3. Scalability testing. Streamlit is lightweight, and we're reading and writing all prompts on refresh.

See also the [design doc](https://docs.google.com/document/d/1IQzrrAAMPS0XAn_ArOq2hyEDCVfeB7AfcvLUqgSnWxQ/).

## Known Issues

**Datasets with configurations:** Loading datasets that require a configuration is currently broken. Need to
either figure out how to query for available options and let user pick, or only use first one or something.
Unclear right now if a template for one configuration can be applied to another in all cases.

**Warning or Error about Darwin on OS X:** Try downgrading PyArrow to 3.0.0.
