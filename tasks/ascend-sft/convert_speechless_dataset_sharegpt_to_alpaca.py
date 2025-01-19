#!/usr/bin/env python

import json
from tqdm import tqdm

def convert(input_file, output_file):
    json_data = [ json.loads(line.strip()) for line in open(input_file, 'r')]
    new_data = []
    for d in tqdm(json_data, desc='Loading'):
       category = d['category'] 
       c = d['conversations']
       assert len(c) == 2
       instruction = c[0]['value']
       output = c[1]['value']
       meta = {'category': category}
       item = {
           'instruction': instruction,
           'input': "",
           'output': output,
           'meta': meta,
       }
       new_data.append(item)

    with open(output_file, 'w') as f:
        for d in tqdm(new_data, desc='Saving'):
            f.write(json.dumps(d, ensure_ascii=False) + '\n')

def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    return parser.parse_args()

def main():
    args = get_args()
    convert(args.input_file, args.output_file)

if __name__ == '__main__':
    main()    