# PromptSource
Toolkit for collecting and applying templates of prompting instances.

WIP

## Setup
1. Download the repo
2. Navigate to root directory of the repo
3. Install requirements with `pip install -r requirements.txt` in a Python 3.7 environment

## Running
From the root directory of the repo, you can launch the editor with
```
streamlit run promptsource/app.py
```

There are 3 modes in the app:
- **Helicopter view**: aggregate high level metrics on the current state of the sourcing
- **Prompted dataset viewer**: check the templates you wrote or already written on entire dataset
- **Sourcing**: write new prompts

<img src="assets/promptsource_app.png" width="800">

## Running (read-only)
To host a public streamlit app, launch it with
```bash
streamlit run promptsource/app.py -- -r
```

## Prompting an Example:
You can use Promptsource with [Datasets](https://huggingface.co/docs/datasets/) to create
prompted examples:
```python
# Get an example
from datasets import load_dataset
dataset = load_dataset("ag_news")
example = dataset["train"][0]

# Prompt it
from promptsource.templates import TemplateCollection
# Get all the prompts
collection = TemplateCollection()
# Get all the AG News prompts
ag_news_prompts = collection.get_dataset("ag_news")
# Select a prompt by name
prompt = ag_news_prompts["classify_question_first"]

result = prompt.apply(example)
print("INPUT: ", result[0])
print("TARGET: ", result[1])
```

## Contributing
Contribution guidelines and step-by-step *HOW TO* are described [here](CONTRIBUTING.md).

## Writing Prompts
A prompt is expressed in [Jinja](https://jinja.palletsprojects.com/en/3.0.x/).

It is rendered using an example from the corresponding Hugging Face datasets library
(a dictionary). The separator ||| should appear once to divide the template into input
and target. Generally, the prompt should provide information on the desired behavior,
e.g., text passage and instructions, and the output should be a desired response.

For more information, read the [Contribution guidelines](CONTRIBUTING.md).

## Known Issues

**Warning or Error about Darwin on OS X:** Try downgrading PyArrow to 3.0.0.

**ConnectionRefusedError: [Errno 61] Connection refused:** Happens occasionally. Try restarting the app.
