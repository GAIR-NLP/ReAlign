# Reformatted Alignment
This is the official repository for [**Reformatted Alignment**](https://arxiv.org/abs/xxx).

## News
- **Jan 2024**: We release the preprint paper on Arxiv, ReAlign data, and other useful resources in developing them (tasks description, hand-written format, tasks classifier, its data, and nq dataset for factuality evaluation).

## Table of contents

- [Introduction](#Introduction)
- [Quick Start](#quick-start)
  - [Setup](#setup)
  - [Pipeline](#pipeline)
- [ReAlign Dataset](#realign-dataset)
- [Other Resources](#other-resources)
  - [Tasks Description and Formats](#scenarios)
  - [Task Classifier: Model and Data](#scenario-classifier)
  - [Factuality Evaluation: NQ Dataset, ChatGPT Responses, and ReAlign Responses](#factuality-evaluation)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

## Introduction

## Quick Start

### Setup
We use `python 3.10` in this project. You are encouraged to create a virtual environment through `conda`.

Then, we have to install all the libraries listed in `requirements.txt`. Note that you may choose an appropriate version of `torch` according to your CUDA version (we write `torch>=2.0.1+cu118` in this file).

```bash
pip install -r requirements.txt
```

### Pipeline
* get your OpenAI API key from [here](https://beta.openai.com/). This is used for reformatting.
* get your Serper API key from [here](https://serper.dev/). This is only used for retrieval with Google Search. 

#### Step 1: Task Classification
Download the task classifier from huggingface hub:

| Model Name      | HF Checkpoints                                                                           | Size    | License                                                      |
|-----------------|------------------------------------------------------------------------------------------| ------- | ------------------------------------------------------------ |
| Task Classifier | [🤗 GAIR/ReAlign-task-classifier](https://huggingface.co/GAIR/) | **13B** | [Llama 2](https://ai.meta.com/resources/models-and-libraries/llama-downloads/) |

Then, by using the following prompt, the task classifier can identify which task a query belongs to:
```python
PROMPT_INPUT_FOR_TASK_CLS: str = '''
You will receive a user's query. Additionally, you are given some pre-defined tasks below: 

[Existing tasks start]
question_generation
story_generation
poem_generation
email_generation
data_generation
advice_giving
recommendations
how_to_generation
planning
instructional_rewriting
language_polishing
paraphrasing
text_correction
code_correction
code_simplification
information_extraction
keywords_extraction
table_extraction
title_generation
text_summarization
note_summarization
explain_code
explain_answer
text_to_text_translation
text_to_code_translation
code_to_code_translation
code_to_text_translation
open_qa
closed_qa
fill_in_the_blank
fact_verification
math_puzzles
language_learning_questions
natural_language_learning_tutor
exam_problem_solving_tutor
ml_ai_language_model_tutor
general_classification
ordering
sentiment_analysis
code_language_classification
language_classification
topic_classification
value_judgement
rejecting
roleplay
default
[Existing tasks end]

You objective is to choose the most appropriate task that can reflect the high-level intention of this query. You should first clearly give out your choice. Your choice should exactly match one of the task names provided above, without any modification. Do not include the task description in your choice.

Your output should be just the task name.

User's query is below:
[User's query start]
{input}
[User's query end]

Task name:

'''
```
Here is an example:
```python
from vllm import LLM, SamplingParams
import torch

num_gpus = torch.cuda.device_count()
model_name_or_dir = "GAIR/ReAlign-task-classifier" # or the local directory to store the downloaded model
llm = LLM(model=model_name_or_dir, tensor_parallel_size=num_gpus)

query = "Give three tips for staying healthy."
input_ = PROMPT_INPUT_FOR_TASK_CLS.format(input=query)

sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=50)
outputs = llm.generate(input_, sampling_params)
task = output[0].outputs[0].text

print(task) # should be `advice_giving`.
# If the classified result is not in task list, set it as `default`.
```

#### Step 2: Prepare your dataset
Convert your dataset into the following format with json type, same as the ReAlign datasets.

Here is an example:
```python
[
    {
        "id": 0,
        "items": [
            {
                # question
                "from": "human",
                "value": "Give three tips for staying healthy.",
                "category": "advice_giving"
            },
            {
                # response
                "from": "gpt",
                "value": "1.Eat a balanced diet and make sure to include plenty of fruits and vegetables. \n2. Exercise regularly to keep your body active and strong. \n3. Get enough sleep and maintain a consistent sleep schedule."
            }
        ]
    }
]
```

#### Step 3: Retrieval with Google Search
Set your Serper API key: 
```python
export SERPER_API_KEY=...
```
Run the following script:
```python
python retrieval.py \
    --input_data_path dataset.json \
    --output_path dataset_retrieval.json \
    --batch_size 10
```
The output file:

`dataset_retrieval.json`is added the original retrieval results.

`dataset_retrieval_clean_evidence.json`is added the cleaned retrieval results.
This is used for ReAlign.

#### Step 4: Reformat
Set your OpenAI API key: 
```python
export OPENAI_API_KEY=...
```


## ReAlign Dataset

## Other Resources

### Tasks Description and Formats

### The Data for Task Classifier

### Factuality Evaluation: NQ Dataset, ChatGPT Responses, and ReAlign Responses

## Citation

## Acknowledgements