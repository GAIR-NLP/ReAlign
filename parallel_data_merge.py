import json
from tqdm import tqdm
from argparse import ArgumentParser
import os
from util import load_json_data, load_jsonl_data


def main():
    parser = ArgumentParser()
    parser.add_argument('--input_data_path', type=str)
    parser.add_argument('--output_directory', type=str)
    parser.add_argument('--final_output_path', type=str)
    args = parser.parse_args()

    filenames = os.listdir(args.output_directory)

    original_data = load_json_data(args.input_data_path)

    print(f"Original data: {len(original_data)} samples.\n")

    rewrite_data = []

    for filename in filenames:
        data_path = os.path.join(args.output_directory, filename)
        data = load_jsonl_data(data_path)
        print(f"Load {len(data)} samples from {data_path}")

        rewrite_data.extend(data)

    print(f"\nTotally load {len(rewrite_data)} rewrite data.")
    if len(original_data) != len(rewrite_data):
        print("The number is error! please check!")
        exit(0)

    rewrite_data = sorted(rewrite_data, key=lambda x: x['id'])

    for data in original_data:
        idx = data['id']
        data['items'][1]['value'] = rewrite_data[idx]['items'][1]['revised_response']['revised_response']

    with open(args.final_output_path, 'w') as f:
        f.write(json.dumps(original_data, indent=4, ensure_ascii=False))


if __name__ == '__main__':
    main()