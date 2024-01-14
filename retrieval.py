import json

from google_serper import GoogleSerperAPIWrapper
from util import load_json_data
from constant import RAG_TASKS
from argparse import ArgumentParser
import math
import os
from tqdm import tqdm
import asyncio

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def retrieval():
    parser = ArgumentParser()
    parser.add_argument('--input_data_path')
    parser.add_argument('--output_path')
    parser.add_argument('--batch_size', type=int, default=10)
    args = parser.parse_args()

    search = GoogleSerperAPIWrapper()

    if os.path.exists(args.output_path):
        original_data = load_json_data(args.output_path)
    else:
        original_data = load_json_data(args.input_data_path)
    logger.info(f'Loaded {len(original_data)} samples')

    need_retrieval_data = [d for d in original_data if d['items'][0]["category"] in RAG_TASKS and len(d['items']) == 2]
    logger.info(f'{len(need_retrieval_data)} samples have retrieval task')

    retrieval_data = []

    for ret_d in need_retrieval_data:
        if ret_d['items'][0].get('evidence'):
            continue
        retrieval_data.append(ret_d)

    logger.info(f'{len(need_retrieval_data) - len(retrieval_data)} samples have been retrieved')
    logger.info(f'{len(retrieval_data)} samples will be retrieved')
    num_batches = math.ceil(len(retrieval_data) / args.batch_size)

    for i in tqdm(range(num_batches)):
        # logger.info(f"Batch {i}")
        batch_start = i * args.batch_size
        batch_end = min((i + 1) * args.batch_size, len(retrieval_data))
        queries = [q['items'][0]["value"] for q in retrieval_data[batch_start:batch_end]]
        responses = asyncio.run(search.run(queries))

        for idx, res in enumerate(responses):
            evidences = []
            for evidence in res:
                evidences.append(evidence['content'])
            retrieval_data[batch_start + idx]['items'][0]['evidence'] = evidences

        for ret_d in retrieval_data[batch_start:batch_end]:
            original_data[ret_d['id']] = ret_d

        with open(args.output_path, 'w') as fp:
            fp.write(json.dumps(original_data, indent=4, ensure_ascii=False))

    # clean evidence
    logger.info("Start to clean evidence.")
    data = load_json_data(args.output_path)
    delet_num = 0
    for d in tqdm(data):
        if d['items'][0]["category"] in RAG_TASKS and d['items'][0].get('evidence'):
            evidences = d['items'][0].get('evidence', [])
            new_evidences = []
            for evidence in evidences:
                if evidence == "No good Google Search Result was found" or \
                        evidence.startswith("Missing:") or \
                        evidence.startswith('Duration:') or evidence.startswith("Posted:"):
                    continue
                new_evidences.append(evidence)
            if len(new_evidences) > 0:
                d['items'][0]['evidence'] = new_evidences
            else:
                if 'evidence' in d['items'][0]:
                    delet_num += 1
                    del d['items'][0]['evidence']

    logger.info(f"delet {delet_num}")

    name, ext = os.path.splitext(args.output_path)
    new_output_path = name + "_clean_evidence" + ext
    with open(new_output_path, 'w') as fp:
        fp.write(json.dumps(data, indent=4, ensure_ascii=False))


def main():
    retrieval()


if __name__ == '__main__':
    main()
