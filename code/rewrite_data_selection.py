import json
from tqdm import tqdm
from argparse import ArgumentParser
import os
from util import load_json_data, load_jsonl_data
import re
from transformers import AutoTokenizer


def math_selection(question, original_response, rewrite_response):
    pattern = '\d*\.?\d+'
    pred = re.findall(pattern, question)
    if len(pred) >= 1:
        return True
    return False


def code_selection(question, original_response, rewrite_response):
    pattern = re.compile(r'```[\s\S]*?```')
    original = bool(pattern.search(original_response))
    rewrite = bool(pattern.search(rewrite_response))

    original_def = 'def ' in original_response and 'return ' in original_response

    rewrite_def = 'def ' in rewrite_response and 'return ' in rewrite_response

    original_code = original or original_def
    rewrite_code = rewrite or rewrite_def

    if original_code and not rewrite_code:
        return False
    if not original_code and rewrite_code:
        return False
    return True


def exam_selection(question, original_response, rewrite_response):
    if len(original_response) <= 5:
        return False
    return True


def open_platypus_filtering(question, response):
    if '### Instruction:' in response or '### Response:' in response:
        return True
    return False

def length_selection(question, original_response, rewrite_response, tokenizer, threshold=0.5):
    original_length = len(tokenizer(original_response)['input_ids']) - 1
    rewrite_length = len(tokenizer(rewrite_response)['input_ids']) - 1
    if rewrite_length <= (threshold * original_length):
        return False
    return True


def open_platypus_overall_selection(question, original_response, rewrite_response, tokenizer):
    if len(original_response.strip()) == 1:
        pattern = rf'(?<![a-zA-Z]){original_response.strip()}(?![a-zA-Z])'
        result = re.search(pattern, rewrite_response)
        if result:
            return True
        else:
            return False
    if original_response.strip() == 'False' or original_response.strip() == 'True':
        return True

    if not code_selection(question, original_response, rewrite_response):
        return False
    else:
        return length_selection(question, original_response, rewrite_response, tokenizer)


def alpaca_overall_selection(question, original_response, rewrite_response, tokenizer):
    if not code_selection(question, original_response, rewrite_response):
        return False
    else:
        return length_selection(question, original_response, rewrite_response, tokenizer)


def no_robots_overall_selection(idx, question, original_response, rewrite_response, tokenizer, category):
    if category == 'default':
        return False
    if 'Revised response' in rewrite_response:
        return False
    if category == 'planning':
        if ' plan' not in question and ' Plan' not in question:
            return False
    if not code_selection(question, original_response, rewrite_response):
        return False
    else:
        return length_selection(question, original_response, rewrite_response, tokenizer, threshold=0.5)


category_selection_rules_mapping = {
    'math_puzzles': math_selection,
    'text_to_code_translation': code_selection,
    'exam_problem_solving_tutor': exam_selection
}


def main():
    parser = ArgumentParser()
    parser.add_argument('--input_original_data_path', type=str)
    parser.add_argument('--input_rewrite_data_path', type=str)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--tokenizer_path', type=str)
    args = parser.parse_args()

    original_data = load_json_data(args.input_original_data_path)
    rewrite_data = load_json_data(args.input_rewrite_data_path)
    new_data = []

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    rew_num = 0
    orginal_num = 0

    for i, d in tqdm(enumerate(original_data)):
        question = d["items"][0]["value"]
        category = d["items"][0]["category"]
        original_response = d["items"][1]["value"]
        rewrite_response = rewrite_data[i]["items"][1]["value"]
        # if no_robots_overall_selection(i, question, original_response, rewrite_response, tokenizer, category):
        # if alpaca_overall_selection(question, original_response, rewrite_response, tokenizer):
        if open_platypus_overall_selection(question, original_response, rewrite_response, tokenizer):
            new_data.append(rewrite_data[i])
            new_data[-1]['tag'] = "rewrite"
            rew_num += 1
        else:
            new_data.append(d)
            new_data[-1]['tag'] = "original"
            orginal_num += 1

    print(f"rewrite num: {rew_num}\noriginal num: {orginal_num}")
    print(len(new_data))

    with open(args.output_path, 'w') as f:
        f.write(json.dumps(new_data, indent=4, ensure_ascii=False))


if __name__ == '__main__':
    main()