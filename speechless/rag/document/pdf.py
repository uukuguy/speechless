#!/usr/bin/env python

import os
from .pdf2txt import PDFDataProcessor, pdf2txt_lines_to_text

def get_pdf_data_by_pdfplumber(pdf_file):
    import pdfplumber
    pdf = pdfplumber.open(pdf_file)
    pdf_pages = []
    for page in pdf.pages:
        page_text = page.extract_words()
        table_settings = {'intersection_x_tolerance': 1}
        tables = page.find_tables(table_settings=table_settings)

        page_tables = []
        for table in tables:
            rows = table.extract()
            page_tables.append({'bbox': table.bbox, 'rows': rows})

        pdf_pages.append({
            'page_no': page.page_number,
            'width': page.width,
            'height': page.height,
            'text': page_text,
            'tables': page_tables,
        })
    pdf_data = {
        'filename': os.path.realpath(pdf_file),
        'metadata': pdf.metadata,
        'pages': pdf_pages,
    }
    return pdf_data


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
            self.pdf_data = get_pdf_data_by_pdfplumber(filepath)

            processor = PDFDataProcessor(pdf_data=self.pdf_data)
            processor.process_pdf()
            all_text = processor.all_text

            self.all_lines = [d for k, d in all_text.items() if 'page' in d]

    def show_text(self):
        full_text = pdf2txt_lines_to_text(self.all_lines)
        all_lines = full_text.split('\n')
        for line in all_lines:
            print(line)

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
