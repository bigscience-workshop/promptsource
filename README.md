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

There are 3 modes in the app:
- **Helicopter view**: aggregate high level metrics on the current state of the sourcing
- **Prompted dataset viewer**: check the templates you wrote or already written on entire dataset
- **Sourcing**: write new prompts

## Writing Templates
A prompt template is expressed in [Jinja](https://jinja.palletsprojects.com/en/3.0.x/).

It is rendered using an example from the corresponding Hugging Face datasets library
(a dictionary). The separator ||| should appear once to divide the template into prompt
and output. Generally, the prompt should provide information on the desired behavior,
e.g., text passage and instructions, and the output should be a desired response.

Here's an example for [AG News](https://huggingface.co/datasets/ag_news):
```
{{text}}
Is this a piece of news regarding world politics, sports, business, or technology? |||
{{ ["World politics", "Sport", "Business", "Technology"][label] }}
```

## Contributing
This is very much a work in progress, and help is needed and appreciated. Anyone wishing to
contribute code can contact Steve Bach for commit access, or submit PRs from forks. Some particular
places you could start:
1. Try to express things! Explore a dataset and tell us what's hard to do to create templates you want
2. Look in the literature. Are there prompt creation methods that do/do not fit well right now?
3. Scalability testing. Streamlit is lightweight, and we're reading and writing all prompts on refresh.

See also the [design doc](https://docs.google.com/document/d/1IQzrrAAMPS0XAn_ArOq2hyEDCVfeB7AfcvLUqgSnWxQ/).

Before submitting a PR or pushing a new commit, please run style formattings and quality checks so that your newly added file look nice:
```bash
make style
make quality
```

## Known Issues

**Warning or Error about Darwin on OS X:** Try downgrading PyArrow to 3.0.0.
