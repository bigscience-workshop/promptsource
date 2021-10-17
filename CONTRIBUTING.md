# Contributing

One of the best ways to contribute is by writing prompts!

### What are Templates?

A prompts is a piece of code written in a templating language called
[Jinja](https://jinja.palletsprojects.com/en/3.0.x/). A prompt defines
a function that maps an example from a dataset in the
[Hugging Face datasets library](https://huggingface.co/datasets) to two strings of
text. The first is called the _input_ which provides all information that
will be available to solve a task, such as the instruction and the context.
The second piece is called the _target_, which is the desired response to the
prompt.

### Quick-Start Guide to Writing Prompts

1. **Set up the app.** Fork the app and set up using the
[README](https://github.com/bigscience-workshop/promptsource/blob/main/README.md).
1. **Examine the dataset.** Select or type the dataset into the dropdown in the app.
If the dataset has subsets (subsets are not the same as splits), you can select
which one to work on. Note that templates are subset-specific. You can find
out background information on the dataset by reading the information in the
app. The dataset is a collection of examples, and each example is a Python
dictionary. The sidebar will tell you the schema that each example has.
1. **Start a new prompt**. Enter a name for your first prompt and hit "Create."
You can always update the name later. If you want to cancel the prompt, select
"Delete Prompt."
1. **Write the prompt**. In the box labeled "Template," enter a Jinja expression.
See the [getting started guide](#getting-started-using-jinja-to-write-prompts)
and [cookbook](#jinja-cookbook) for details on how to write templates.
1. **Add a reference.** If your template was inspired by a paper, note the
reference in the "Prompt Reference" section. You can also add a description of
what your template does.
1. **Save the prompt**. Hit the "Save" button. The output of the prompt
applied to the current example will appear in the right sidebar.
1. **Verify the prompt**. Check that you didn't miss any case by scrolling
through a handful of examples of the prompted dataset using the
"Prompted dataset viewer" mode.
1. **Write between 5 and 10 prompts**. Repeat the steps 4 to 8 to create between 5
and 10 (more if you want!) templates per dataset/subset. Feel free to introduce
a mix of formats, some that follow the templates listed in the [best practices](#best-practices)
and some that are more diverse in the format and the formulation. 
1. **Duplicate the prompts(s).** If the dataset you have chosen bear the same
format as other datasets (for instance, `MNLI` and `SNLI` have identical formats),
you can simply duplicate the prompts you have written to these additional datasets.
1. **Upload the template(s).** Submit a PR using the instructions
[here](#uploading-prompts).

## Getting Started Using Jinja to Write Prompts

Here is a quick crash course on using [Jinja](https://jinja.palletsprojects.com/en/3.0.x/)
to write templates. More advanced usage is in the [cookbook](#jinja-cookbook).

Generally, in a prompt, you'll want to use a mix of hard-coded data that is
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
You can leave binary options like yes/no, true/false, etc. unprotected.

Finally, remember that a template must produce two strings: a prompt and an output.
To separate these two pieces, use three vertical bars `|||`.
So, a complete template for AG News could be:
```jinja2
{{text}}
Is this a piece of news regarding {{"world politics"}}, {{"sports"}}, {{"business"}}, or {{"science and technology"}}? |||
{{ ["World politics", "Sports", "Business", "Science and technology"][label] }}
```

## Best Practices


* **Writing outputs.** When writing a template for a task that requires outputting
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
* **Task Template Checkbox** While there are many different ways to prompt the tasks, only some of them correspond to the original intention of the dataset. For instance, for a summary dataset you can generate a summary or hallucinate an article. However, only the first was the true original task for the dataset. Use the *Task template* check box to indicate true task prompts. (We realize there are some corner cases, for instance, if there was no original task, you should leave this blank. If there are multiple original tasks you can check it for each of them. If you are confused for your dataset, consult with us in slack.) 
* **Filtering templates.** If a template is applied to an example and produces an
empty string, that template/example pair will be skipped. (Either the entire output
is whitespace or the text on either side of the separator `|||` is whitespace.
You can therefore create templates that only apply to a subset of the examples by
wrapping them in Jinja if statements. For example, in the `TREC` dataset, there
are fine-grained categories that are only applicable to certain coarse-grained categories.
We can capture this with the following template:
```jinja2
{% if label_coarse == 0 %}
Is this question asking for a {{"definition"}}, a {{"description"}}, a {{"manner of action"}}, or a {{"reason"}}?
{{text}}
|||
{{ {0: "Manner", 7: "Defintion", 9: "Reason", 12: "Description"}[label_fine] }}
{% endif %}
```
* **Conditional generation format.** Always specify the output label `y` and separate it from the prompt
by indicating the vertical bars `|||`. The `y` will be generated by a generative model
conditioned on the prompted input you wrote. You can always transform an "infix" prompt format
```jinja2
Given that {{premise}}, it {{ ["must be true", "might be true", "must be false"][label] }} that {{hypothesis}}
```
into a conditional generation format
```jinja2
Given that {{premise}}, it {{ "must be true, might be true, or must be false" }} that {{hypothesis}}?|||
{{ ["must be true", "might be true", "must be false"][label] }}
```
* **Pre-defined formats.** The goal is to collect a diverse set of prompts with diverse formats, but
we also want to include a few less diverse prompts that follow the following two structures:
1) A question-answer pair with optional multiple choices like:
```
[Context]                         # optional depending on the task
[Question]
[Label1], [Label2], [Label3]      # optional depending on the task
```
So for SNLI it will look like:
```jinja2
{{premise}}
Is it the case that {{hypothesis}}?
{{ "Yes" }}, {{ "No" }}, {{ "Maybe" }} ||| {{ ["Yes", "No", "Maybe"][label] }}
```

2) Task description followed by the input. So for SNLI it will look like:
```jinja2
Determine the relation between the following two sentences. The relations are entailment, contradiction, or neutral.
{{premise}}
{{hypothesis}} ||| {{label}}
```

## More Examples

Here are a few interesting examples of templates with explanations.

Here's one for `hellaswag`:
```jinja2
First, {{ ctx_a.lower() }} Then, {{ ctx_b.lower() }}...

Complete the above description with a chosen ending:

Ending 1: {{ endings[0] }}

Ending 2: {{ endings[1] }}

Ending 3: {{ endings[2] }}

Ending 4: {{ endings[3] }}

||| {{ {"0": "Ending 1", "1": "Ending 2", "2": "Ending 3", "3": "Ending 4"}[label] }}
```
Notice how it uses functions to consistently capitalize the information and provides lots
of context (referring explicitly to "description" and "chosen ending.")

Here's one for `head_qa`:
```jinja2
Given this list of statements about {{category}}: {{ answers | map(attribute="atext")
| map("lower") | map("trim", ".") | join(", ") }}.
Which one is the most appropriate answer/completion for the paragraph that follows?
{{qtext}}
|||
{% for answer in answers if answer["aid"]==ra -%}
{{answer["atext"]}}
{%- endfor %}
```
Like above, it uses functions to present the choices in a readable way. Also, it
uses a for loop with conditions to handle the more intricate dataset schema. 

Here's one for `paws`:
```jinja2
{% if label == 0 or label == 1 %} 
Sentence 1: {{sentence1}}
Sentence 2: {{sentence2}}
Question: Does Sentence 1 paraphrase Sentence 2? Yes or No?
{% endif %}
||| 
{% if label == 0 %} 
No
{% elif label == 1 %}
Yes
{% endif %}

```
This template has to do a few things, even though it's a yes no question. First,
the label might be unknown, so the pieces are wrapped in if statements.
Second, notice that the choices `Yes or No` are not escaped. Yes/no, true/false
are choices that do not need to be escaped (unlike categories).

## Uploading Prompts

Once you save or modify a template, the corresponding file inside the `templates`
directory in the repo will be modified. To upload it, follow these steps:
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
```jinja=
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
- Using `zip()` to zip multiple lists
```jinja
{% for a, b in zip(list_A, list_B) %}
do_something_with_a_and_b
{% endfor %} 
```


Jinja includes lots of complex features but for most instances you likely only
need to use the methods above. If there's something you're not sure how to do,
just message the prompt engineering group on Slack. We'll collect other frequent
patterns here.
