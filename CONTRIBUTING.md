# Contributing

One of the best ways to contribute is by writing templates!

### What are Templates?

A template is a piece of code written in a templating language called
[Jinja](https://jinja.palletsprojects.com/en/3.0.x/). A template defines
a function that maps an example from a dataset in the
[HuggingFace library](https://huggingface.co/datasets) to two strings of
text. The first is called the _prompt_ which provides all information that
will be available to solve a task, such as the instruction and the context.
The second piece is called the _output_, which is the desired response to the
prompt.

### Quick-Start Guide to Writing Templates

1. **Set up the app.** Fork the app and set up using the
[README](https://github.com/bigscience-workshop/promptsource/blob/main/README.md).
1. **Select a dataset.** Go to the tracking spreadsheet
[here](https://docs.google.com/spreadsheets/d/10SBt96nXutB49H52PV2Lvne7F1NvVr_WZLXD8_Z0JMw/)
and find an unclaimed one. Put your name under "Who's Prompting it?" and
mark it yellow to show it's in progress.
1. **Examine the dataset.** Select or type the dataset into the dropdown in the app.
If the dataset has subsets (subsets are not the same as splits), you can select
which one to work on. Note that templates are subset specific. You can find
out background information on the dataset by reading the information in the
app. The dataset is a collection of examples, and each example is a Python
dictionary. The sidebar will tell you the schema that each example has.
1. **Start a new template**. Enter a name for your first template and hit "Create."
You can always update the name later. If you want to cancel the template, select
"Delete Template."
1. **Write the template**. In the box labeled "Template," enter a Jinja expression.
See the [getting started guide](#getting-started-using-jinja-to-write-templates)
and [cookbook](#jinja-cookbook) for details on how to write templates.
1. **Add a reference.** If your template was inspired by a paper, note the
reference in the "Template Reference" section. You can also add a description of
what your template does.
1. **Save the template**. Hit the "Save" button. The output of the template
applied to the current example will appear in the right sidebar.
1. **Verify the template**. Check that you didn't miss any case by scrolling
through a handful of examples of the prompted dataset using the
"Prompted dataset viewer" mode.
1. **Duplicate the template(s).** If the dataset you have chosen bear the same
format as other datasets (for instance `MNLI` and `SNLI` have identical format),
you can simply claim these datasets and duplicate the templates you have written
to these additional datasets.
1. **Upload the template(s).** Submit a PR using the instructions
[here](#uploading-templates).

## Getting Started Using Jinja to Write Templates

Here is a quick crash course on using [Jinja](https://jinja.palletsprojects.com/en/3.0.x/)
to write templates. More advanced usage is in the [cookbook](#jinja-cookbook).

Generally in a template, you'll want to use a mix of hard-coded data that is
task-specific and stays the same across examples, and commands that tailor the
prompt and output to a specific example.

To write text that should be rendered as written, just write it normally. The
following "template" will produce the same text every time:
```jinja2
This is just literal text that will be printed the same way every time.
```

To make your template do something more interesting, you'll need to use Jinja
expressions. Jinja expressions are surrounded by curly braces `{` and `}`.
One common thing you'll want to do is access information in the dataset example.
When applied to an example, you can access any value in the example dictionary
via its key. If you just want to print that value surround it in double curly
braces. For example, if you want to print a value with the key `text`, use this:
```jinja2
The text in this example is {{ text }}.
```

You can also use information from the example to control behavior. For example,
suppose we have a label with the key `label` in our example, which either has a
value of 0 or 1. That's not very "natural" language, so maybe we want to decide
which label name to use based on the example. We can do this by creating a list
and indexing it with the example key:
```jinja2
The label for this example is {{ ["Label A", "Label B"][label] }}.
```
We can also use dictionaries for the same thing:
```jinja2
The label for this example is {{
{"a": "Label A",
 "b": "Label B"
}[label]
}}.
```

Note that some things in a template are particular to the task, and should not be
modified by downstream steps that try to increase the diversity of the prompts.
A common example is listing label names in the prompt to provide choices. Anything
that should not be modified by data augmentation should be surrounded by double
curly braces and quoted. For example:
```jinja2
The choices are {{"a"}}, {{"b"}}, and {{"c"}}.
```

Finally, remember that a template must produce two strings: a prompt and an output.
To separate these two pieces, use three vertical bars `|||`.
So, a complete template for AG News could be:
```jinja2
{{text}}
Is this a piece of news regarding {{"world politics"}}, {{"sports"}}, {{"business"}}, or {{"science and technology"}}? |||
{{ ["World politics", "Sports", "Business", "Science and technology"][label] }}
```

## Best Practices

A few miscellaneous things:

* **Writing outputs.** When writing a template for an task that requires outputting
a label, don't use articles or other stop words before the label name in the output.
For example, in TREC, the output should be "Person", not "A person". The reason
is that evaluations often look at the first word of the generated output to determine
correctness.
*  **Skipping datasets.** You might find a dataset in the spreadsheet that it
doesn't make sense to write templates for. For example, a dataset might just be
text without any annotations. For other cases, ask on Slack. If skipping a dataset,
mark it in red on the spreadsheet.
* **Choosing input and output pairs.** Lots of datasets have multiple columns that can be
combined to form different (input, output) pairs i.e. different "tasks". Don't hesitate to
introduce some diversity by prompting a given dataset into multiple tasks and provide some
description in the "Template Reference" text box. An example is given
in the already prompted `movie_rationales`.
* **Filtering templates.** If a template is applied to an example and produces an
empty string, that template/example pair will be skipped. You can therefore create
templates that only apply to a subset of the examples by wrapping them in Jinja
if statements. For example, in the `TREC` dataset, there are fine-grained
categories that are only applicable to certain coarse-grained categories. We can
capture this with the following template:
```jinja2
{% if label_coarse == 0 %}
Is this question asking for a {{"definition"}}, a {{"description"}}, a {{"manner of action"}}, or a {{"reason"}}?
{{text}}
|||
{{ {0: "Manner", 7: "Defintion", 9: "Reason", 12: "Description"}[label_fine] }}
{% endif %}
```

## Uploading Templates

Once you save or modify a template, the corresponding file inside the `templates`
directory in the repo will be modified. To upload it, following these steps:
1. Run `make style` and `make quality`.
2. Commit the modified template files (anything under `templates`) to git.
3. Push to your fork on GitHub.
4. Open a pull request against `main` on the PromptSource repo.
5. When the PR is merged into main, mark the dataset in green on the spreadsheet.


## Jinja Cookbook

- Accessing nested attributes of a dict
```jinja
{{ answers_spans.spans }}
```

- Joining list
```jinja
{{ spans_list | join(", ") }}
```

- If conditions
```jinja
{% if label==0 %}
do_something
{% elif condition %}
do_something_else
{% endif %}
```

Jinja includes lots of complex features but for most instances you likely only
need to use the methods above. If there's something you're not sure how to do,
just message the prompt engineering group on Slack. We'll collect other frequent
patterns here.