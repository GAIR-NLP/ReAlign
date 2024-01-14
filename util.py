import json
import openai
import asyncio
from typing import Any


def load_single_jsonl_file(data_file):
    with open(data_file)as f:
        lines = f.readlines()
    return [json.loads(l, strict=False) for l in lines]


def load_single_json_file(data_file):
    with open(data_file)as f:
        data = json.load(f)
    return data


def load_jsonl_data(data_file):
    raw_dataset = []
    if isinstance(data_file, str):
        raw_dataset += load_single_jsonl_file(data_file)
    elif isinstance(data_file, list):
        for f_ in data_file:
            raw_dataset += load_single_jsonl_file(f_)
    return raw_dataset


def load_json_data(data_file):
    raw_dataset = []
    if isinstance(data_file, str):
        raw_dataset += load_single_json_file(data_file)
    elif isinstance(data_file, list):
        for f_ in data_file:
            raw_dataset += load_single_json_file(f_)
    return raw_dataset


def ChatGPT_API(system_instruction, user_input, model, api_key, temperature=0.7, top_p=0.9, n=1, target_length=1024):
    openai.api_key = api_key
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": 'user', "content": user_input}
        ],
        temperature=temperature,
        max_tokens=target_length,
        top_p=top_p,
        n=n,
    )
    return response['choices']
