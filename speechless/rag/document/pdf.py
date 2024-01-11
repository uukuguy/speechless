#!/usr/bin/env python

import os
from .pdf2txt import PDFProcessor, PDFDataProcessor, pdf2txt_lines_to_text

class PDF:
    def __init__(self, filepath=None):
        self.filepath = filepath
        self.all_lines = []
        if filepath is not None:
            self.parse(filepath)

    def parse(self, filepath):
        assert filepath is not None
        self.filepath = filepath
        if not os.path.exists(filepath):
            print(f"The file {filepath} does not exist.")
        else:
            processor = PDFProcessor(filepath=filepath)
            processor.process_pdf()
            all_text = processor.all_text

            self.all_lines = [d for k, d in all_text.items() if 'page' in d]

def get_args():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--filepath', type=str, default=None, help='Path to the PDF file to parse.')

    args = parser.parse_args()
    return args

def main():
    args = get_args()
    pdf = PDF(filepath=args.filepath)
    for line in pdf.all_lines:
        if line['type'] == 'text':
            print(line['inside'])
        elif line['type'] == 'excel':
            print(line['inside'])
        else:
            print("")

if __name__ == '__main__':
    main()