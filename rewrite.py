import json
from tqdm import tqdm
from argparse import ArgumentParser
import os
from util import load_json_data, load_jsonl_data
from transformers import AutoTokenizer
import openai
from constant import *
from util import ChatGPT_API
import time

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def _split(a, n):
    # Split list a to n chunks
    # https://stackoverflow.com/questions/2130016/splitting-a-list-into-n-parts-of-approximately-equal-length
    k, m = divmod(len(a), n)
    return [a[i*k+min(i, m): (i+1)*k+min(i+1, m)] for i in range(n)]


def select_optimal_response(response_list, tokenizer):
    optimal_response = None
    optimal_len = 0
    for response in response_list:
        length = len(tokenizer(response['revised_response'])['input_ids'])
        if length > optimal_len:
            optimal_response = response
            optimal_len = length
    return optimal_response


def evidence_str_generate(evidence_list):
    evidence_str = ""
    for i, evidence in enumerate(evidence_list):
        evidence_str = evidence_str + f"{i+1}. {evidence}\n\n"
    evidence_str = evidence_str.strip()
    return evidence_str


def rewrite_response_request(question, response, structure, evidences, args):
    if len(evidences) > 0:
        system_instruction = REWRITING_RETRIEVAL_SYSTEM_PROMPT
        evidence_str = evidence_str_generate(evidences)
        user_input = REWRITING_RETRIEVAL_USER_PROMPT.format(question=question, response=response, structure=structure, evidence=evidence_str)
    else:
        system_instruction = REWRITING_SYSTEM_PROMPT
        user_input = REWRITING_USER_PROMPT.format(question=question, response=response, structure=structure)

    rewrited_response = ChatGPT_API(system_instruction.strip(), user_input.strip(), args.model, args.openai_key,
                                    temperature=args.temperature, top_p=args.top_p, n=args.top_k,
                                    target_length=args.target_length)
    rewrited_response = [res['message']['content'] for res in rewrited_response]
    new_response = []
    for i, _ in enumerate(rewrited_response):
        reason, revised_response = _.split('Revised response:', 1)
        reason = reason.split('Reasoning:', 1)[1].strip()
        new_response.append(
            {
                'reason': reason.strip(),
                'revised_response': revised_response.strip()
            }
        )
    return new_response


def main():
    parser = ArgumentParser()
    parser.add_argument('--input_data_path', type=str)
    parser.add_argument('--output_directory', type=str)
    parser.add_argument('--tokenizer_path', type=str)
    parser.add_argument('--dataset_batch_id', type=int, default=0)
    parser.add_argument('--dataset_batch_num', type=int, default=10)
    parser.add_argument('--openai_key', type=str)
    parser.add_argument('--is_retrieval', action='store_true', default=True)
    parser.add_argument('--top_k', type=int, default=2)
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo-1106')
    parser.add_argument('--temperature', type=float, default=0.3)
    parser.add_argument('--top_p', type=float, default=1)
    parser.add_argument('--target_length', type=int, default=2048)
    args = parser.parse_args()

    logger.info("Parameters Info:")

    for k, v in vars(args).items():
        print('{key}: {value}'.format(key=k, value=v))

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    os.environ['OPENAI_API_KEY'] = args.openai_key

    overall_data = load_json_data(args.input_data_path)
    logger.info(f"Overall {len(overall_data)} samples.")
    current_data = _split(overall_data, args.dataset_batch_num)[args.dataset_batch_id]
    logger.info(f"Currently execute the {args.dataset_batch_id} batch, totally containing{len(current_data)} samples.")
    args.output_path = os.path.join(args.output_directory, f"rewrite_data_{args.dataset_batch_id}_of_{args.dataset_batch_num}_batch.jsonl")
    logger.info(f"Output file path: {args.output_path}")

    if os.path.exists(args.output_path):
        completed_data = load_jsonl_data(args.output_path)
        completed_num = len(completed_data)
    else:
        completed_num = 0
    process_data = current_data[completed_num:]
    logger.info(f'Completed {completed_num} / {len(current_data)}')
    logger.info(f'{len(process_data)} will be processed.')

    for data in tqdm(process_data):
        question = data["items"][0]['value']
        category = data["items"][0]['category']
        structure = STRUCTURE[category].strip()
        response = data["items"][1]["value"]
        if args.is_retrieval:
            evidences = data["items"][0].get("evidence", [])
        else:
            evidences = []
        if category not in REWRITE_TASK:
            with open(args.output_path, 'a+') as writer:
                writer.write(json.dumps(data, ensure_ascii=False) + '\n')
            continue
        while True:
            try:
                new_response_list = rewrite_response_request(question, response, structure, evidences, args)
                if len(new_response_list) == 0:
                    logger.info("response cannot be parsed")
                    continue
            except Exception as e:
                logger.info(e)
                logger.info("retry")
                time.sleep(1)
                continue
            break
        data["items"][1]['revised_response_list'] = new_response_list
        if len(new_response_list) > 1:
            new_response = select_optimal_response(new_response_list, tokenizer)
        else:
            new_response = new_response_list[0]
        data["items"][1]['revised_response'] = new_response

        with open(args.output_path, 'a+') as writer:
            writer.write(json.dumps(data, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
