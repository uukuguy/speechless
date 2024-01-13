#!/usr/bin/env python
# https://github.com/MetaGLM/FinGLM/tree/main/tools/pdf_to_txt
import os, glob
import pdfplumber
import re
from typing import Dict
from collections import defaultdict
import json

def resort_pdf_data_text(pdf_data: Dict):
    pages = pdf_data['pages']
    for page in pages:
        page_text_lines = page['text']
        for text_line in page_text_lines:
            """
            # pkl_files/0c9423f0a75af5b150600422e165e71e4df7edaf0d68a57c4456a3c76bb16e57.pdf.pkl

              {'text': '当期非经常性损益明细表', 'x0': 111.02, 'x1': 227.06384000000003, 'top': 301.65895999999987, 'doctop': 158617.4189600003, 'bottom': 312.2189599999999, 'upright': True, 'direction': 1},

              {'text': '√适用', 'x0': 89.904, 'x1': 121.584, 'top': 318.2189599999999, 'doctop': 158633.9789600003, 'bottom': 328.77896, 'upright': True, 'direction': 1},
              {'text': '□不适用', 'x0': 132.02, 'x1': 174.02768, 'top': 318.2189599999999, 'doctop': 158633.9789600003, 'bottom': 328.77896, 'upright': True, 'direction': 1},
                        {'text': '单位：元币种：人民币', 'x0': 426.55, 'x1': 531.7038399999999, 'top': 331.89896, 'doctop': 158647.6589600003, 'bottom': 342.45896, 'upright': True, 'direction': 1},
            {'text': '项目', 'x0': 178.46, 'x1': 199.58, 'top': 345.93895999999995, 'doctop': 158661.6989600003, 'bottom': 356.49895999999995, 'upright': True, 'direction': 1},
            {'text': '金额', 'x0': 338.11, 'x1': 359.23, 'top': 345.93895999999995, 'doctop': 158661.6989600003, 'bottom': 356.49895999999995, 'upright': True, 'direction': 1},
            {'text': '说明', 'x0': 459.82, 'x1': 480.94, 'top': 345.93895999999995, 'doctop': 158661.6989600003, 'bottom': 356.49895999999995, 'upright': True, 'direction': 1},

            {'text': '非流动资产处置损益', 'x0': 95.544, 'x1': 190.23552, 'top': 360.09896, 'doctop': 158675.8589600003, 'bottom': 370.65896, 'upright': True, 'direction': 1},
                {'text': '-37,751.77', 'x0': 351.43, 'x1': 404.11384, 'top': 360.09896, 'doctop': 158675.8589600003, 'bottom': 370.65896, 'upright': True, 'direction': 1},
                    {'text': '资产报废损失', 'x0': 414.79, 'x1': 477.91768, 'top': 360.09896, 'doctop': 158675.8589600003, 'bottom': 370.65896, 'upright': True, 'direction': 1},

            {'text': '计入当期损益的政府补助（与企业业务密', 'x0': 95.544, 'x1': 282.76224, 'top': 374.25896, 'doctop': 158690.0189600003, 'bottom': 384.81896, 'upright': True, 'direction': 1},
                    {'text': '政府补贴见第十节财务', 'x0': 414.79, 'x1': 525.9445599999999, 'top': 380.97896, 'doctop': 158696.7389600003, 'bottom': 391.53896, 'upright': True, 'direction': 1},
            {'text': '切相关，按照国家统一标准定额或定量享', 'x0': 95.544, 'x1': 282.7728, 'top': 387.81896, 'doctop': 158703.5789600003, 'bottom': 398.37896, 'upright': True, 'direction': 1},

                {'text': '7,257,356.31', 'x0': 340.99, 'x1': 404.11768, 'top': 387.81896, 'doctop': 158703.5789600003, 'bottom': 398.37896, 'upright': True, 'direction': 1},
                    {'text': '报告“七、84政府补助”', 'x0': 414.79, 'x1': 520.99408, 'top': 394.65896, 'doctop': 158710.4189600003, 'bottom': 406.36999999999995, 'upright': True, 'direction': 1},

            {'text': '受的政府补助除外）', 'x0': 95.544, 'x1': 190.23552, 'top': 401.49895999999995, 'doctop': 158717.2589600003, 'bottom': 412.05895999999996, 'upright': True, 'direction': 1},

            {'text': '除上述各项之外的其他营业外收入和支', 'x0': 95.544, 'x1': 282.62784, 'top': 415.53896, 'doctop': 158731.2989600003, 'bottom': 426.09896, 'upright': True, 'direction': 1},
                    {'text': '-219,646.25', 'x0': 346.27, 'x1': 404.11768, 'top': 422.37895999999995, 'doctop': 158738.1389600003, 'bottom': 432.93895999999995, 'upright': True, 'direction': 1},
            {'text': '出', 'x0': 95.544, 'x1': 106.104, 'top': 429.21896, 'doctop': 158744.9789600003, 'bottom': 439.77896, 'upright': True, 'direction': 1},

            {'text': '其他符合非经常性损益定义的损益项目', 'x0': 95.544, 'x1': 274.25088, 'top': 443.27896, 'doctop': 158759.0389600003, 'bottom': 453.83896, 'upright': True, 'direction': 1},
                {'text': '26,088,856.31', 'x0': 335.71, 'x1': 404.11767999999995, 'top': 443.27896, 'doctop': 158759.0389600003, 'bottom': 453.83896, 'upright': True, 'direction': 1},

                    {'text': '进项税加计抵减', 'x0': 414.79, 'x1': 488.47768, 'top': 443.27896, 'doctop': 158759.0389600003, 'bottom': 453.83896, 'upright': True, 'direction': 1},
            {'text': '减：所得税影响额', 'x0': 95.544, 'x1': 179.67552, 'top': 457.43895999999995, 'doctop': 158773.1989600003, 'bottom': 467.99895999999995, 'upright': True, 'direction': 1},
                {'text': '5,053,431.36', 'x0': 340.99, 'x1': 404.11768, 'top': 457.43895999999995, 'doctop': 158773.1989600003, 'bottom': 467.99895999999995, 'upright': True, 'direction': 1},

                {'text': '合计', 'x0': 178.46, 'x1': 199.58, 'top': 471.59896, 'doctop': 158787.3589600003, 'bottom': 482.15896, 'upright': True, 'direction': 1},
                {'text': '28,035,383.24', 'x0': 335.71, 'x1': 404.11767999999995, 'top': 471.59896, 'doctop': 158787.3589600003, 'bottom': 482.15896, 'upright': True, 'direction': 1},
            """
            pass

# class PDFProcessor:
#     def __init__(self, filepath=None, pdf=None):
#         assert filepath is not None or pdf is not None
#         if filepath is not None:
#             self.filepath = filepath
#             self.pdf = pdfplumber.open(filepath)
#         else:
#             self.pdf = pdf
#         self.all_text = defaultdict(dict)
#         self.allrow = 0
#         self.last_num = 0

#     def check_lines(self, page, top, buttom):
#         lines = page.extract_words()[::]
#         text = ''
#         last_top = 0
#         last_check = 0
#         for l in range(len(lines)):
#             each_line = lines[l]
#             check_re = '(?:。|；|单位：元|单位：万元|币种：人民币|\d|报告(?:全文)?(?:（修订版）|（修订稿）|（更正后）)?)$'
#             if top == '' and buttom == '':
#                 if abs(last_top - each_line['top']) <= 2:
#                     text = text + each_line['text']
#                 elif last_check > 0 and (page.height * 0.9 - each_line['top']) > 0 and not re.search(check_re, text):

#                     text = text + each_line['text']
#                 else:
#                     text = text + '\n' + each_line['text']
#             elif top == '':
#                 if each_line['top'] > buttom:
#                     if abs(last_top - each_line['top']) <= 2:
#                         text = text + each_line['text']
#                     elif last_check > 0 and (page.height * 0.85 - each_line['top']) > 0 and not re.search(check_re,
#                                                                                                           text):
#                         text = text + each_line['text']
#                     else:
#                         text = text + '\n' + each_line['text']
#             else:
#                 if each_line['top'] < top and each_line['top'] > buttom:
#                     if abs(last_top - each_line['top']) <= 2:
#                         text = text + each_line['text']
#                     elif last_check > 0 and (page.height * 0.85 - each_line['top']) > 0 and not re.search(check_re,
#                                                                                                           text):
#                         text = text + each_line['text']
#                     else:
#                         text = text + '\n' + each_line['text']
#             last_top = each_line['top']
#             last_check = each_line['x1'] - page.width * 0.85

#         return text

#     def drop_empty_cols(self, data):
#         # 删除所有列为空数据的列
#         transposed_data = list(map(list, zip(*data)))
#         # filtered_data = [col for col in transposed_data if not all(cell is '' for cell in col)]
#         filtered_data = [col for col in transposed_data if not all(cell == '' for cell in col)]
#         result = list(map(list, zip(*filtered_data)))
#         return result

#     def extract_text_and_tables(self, page):
#         buttom = 0
#         # FIXME
#         tables = page.find_tables(table_settings={'intersection_x_tolerance': 1})
#         if len(tables) >= 1:
#             count = len(tables)
#             for table in tables:
#                 if table.bbox[3] < buttom:
#                     pass
#                 else:
#                     count -= 1
#                     top = table.bbox[1]
#                     text = self.check_lines(page, top, buttom)
#                     text_list = text.split('\n')
#                     for _t in range(len(text_list)):
#                         self.all_text[self.allrow] = {'page': page.page_number, 'allrow': self.allrow,
#                                                       'type': 'text', 'inside': text_list[_t]}
#                         self.allrow += 1

#                     buttom = table.bbox[3]
#                     new_table = table.extract()
#                     r_count = 0
#                     for r in range(len(new_table)):
#                         row = new_table[r]
#                         if row[0] is None:
#                             r_count += 1
#                             for c in range(len(row)):
#                                 if row[c] is not None and row[c] not in ['', ' ']:
#                                     if new_table[r - r_count][c] is None:
#                                         new_table[r - r_count][c] = row[c]
#                                     else:
#                                         new_table[r - r_count][c] += row[c]
#                                     new_table[r][c] = None
#                         else:
#                             r_count = 0

#                     end_table = []
#                     for row in new_table:
#                         if row[0] != None:
#                             cell_list = []
#                             cell_check = False
#                             for cell in row:
#                                 if cell != None:
#                                     cell = cell.replace('\n', '')
#                                 else:
#                                     cell = ''
#                                 if cell != '':
#                                     cell_check = True
#                                 cell_list.append(cell)
#                             if cell_check == True:
#                                 end_table.append(cell_list)
#                     end_table = self.drop_empty_cols(end_table)

#                     for row in end_table:
#                         self.all_text[self.allrow] = {'page': page.page_number, 'allrow': self.allrow,
#                                                       'type': 'excel', 'inside': str(row)}
#                         # self.all_text[self.allrow] = {'page': page.page_number, 'allrow': self.allrow, 'type': 'excel',
#                         #                               'inside': ' '.join(row)}
#                         self.allrow += 1

#                     if count == 0:
#                         text = self.check_lines(page, '', buttom)
#                         text_list = text.split('\n')
#                         for _t in range(len(text_list)):
#                             self.all_text[self.allrow] = {'page': page.page_number, 'allrow': self.allrow,
#                                                           'type': 'text', 'inside': text_list[_t]}
#                             self.allrow += 1

#         else:
#             text = self.check_lines(page, '', '')
#             text_list = text.split('\n')
#             for _t in range(len(text_list)):
#                 self.all_text[self.allrow] = {'page': page.page_number, 'allrow': self.allrow,
#                                               'type': 'text', 'inside': text_list[_t]}
#                 self.allrow += 1

#         first_re = '[^计](?:报告(?:全文)?(?:（修订版）|（修订稿）|（更正后）)?)$'
#         end_re = '^(?:\d|\\|\/|第|共|页|-|_| ){1,}'
#         if self.last_num == 0:
#             try:
#                 first_text = str(self.all_text[1]['inside'])
#                 end_text = str(self.all_text[len(self.all_text) - 1]['inside'])
#                 if re.search(first_re, first_text) and not '[' in end_text:
#                     self.all_text[1]['type'] = '页眉'
#                     if re.search(end_re, end_text) and not '[' in end_text:
#                         self.all_text[len(self.all_text) - 1]['type'] = '页脚'
#             except:
#                 print(page.page_number)
#         else:
#             try:
#                 first_text = str(self.all_text[self.last_num + 2]['inside'])
#                 end_text = str(self.all_text[len(self.all_text) - 1]['inside'])
#                 if re.search(first_re, first_text) and '[' not in end_text:
#                     self.all_text[self.last_num + 2]['type'] = '页眉'
#                 if re.search(end_re, end_text) and '[' not in end_text:
#                     self.all_text[len(self.all_text) - 1]['type'] = '页脚'
#             except:
#                 print(page.page_number)

#         self.last_num = len(self.all_text) - 1


#     def process_pdf(self):
#         for i in range(len(self.pdf.pages)):
#             self.extract_text_and_tables(self.pdf.pages[i])

#     def save_all_text(self, path):
#         for key in self.all_text.keys():
#             with open(path, 'a+', encoding='utf-8') as file:
#                 file.write(json.dumps(self.all_text[key], ensure_ascii=False) + '\n')

class PDFDataProcessor:
    def __init__(self, pdf_data: Dict):
        self.pdf_data = pdf_data
        self.pages = pdf_data['pages']

        self.all_text = defaultdict(dict)
        self.allrow = 0
        self.last_num = 0

    def check_lines(self, page: Dict, top, buttom):
        # FIXME
        # lines = page.extract_words()[::]
        # page_width = page.width
        # page_height = page.height
        lines = page['text']
        page_width = page['width']
        page_height = page['height']

        text = ''
        last_top = 0
        last_check = 0
        for l in range(len(lines)):
            each_line = lines[l]
            check_re = '(?:。|；|单位：元|单位：万元|币种：人民币|\d|报告(?:全文)?(?:（修订版）|（修订稿）|（更正后）)?)$'
            if top == '' and buttom == '':
                if abs(last_top - each_line['top']) <= 2:
                    text = text + each_line['text']
                elif last_check > 0 and (page_height * 0.9 - each_line['top']) > 0 and not re.search(check_re, text):

                    text = text + each_line['text']
                else:
                    text = text + '\n' + each_line['text']
            elif top == '':
                if each_line['top'] > buttom:
                    if abs(last_top - each_line['top']) <= 2:
                        text = text + each_line['text']
                    elif last_check > 0 and (page_height * 0.85 - each_line['top']) > 0 and not re.search(check_re,
                                                                                                          text):
                        text = text + each_line['text']
                    else:
                        text = text + '\n' + each_line['text']
            else:
                if each_line['top'] < top and each_line['top'] > buttom:
                    if abs(last_top - each_line['top']) <= 2:
                        text = text + each_line['text']
                    elif last_check > 0 and (page_height * 0.85 - each_line['top']) > 0 and not re.search(check_re,
                                                                                                          text):
                        text = text + each_line['text']
                    else:
                        text = text + '\n' + each_line['text']
            last_top = each_line['top']
            last_check = each_line['x1'] - page_width * 0.85

        return text

    def drop_empty_cols(self, data):
        # 删除所有列为空数据的列
        transposed_data = list(map(list, zip(*data)))
        # filtered_data = [col for col in transposed_data if not all(cell is '' for cell in col)]
        filtered_data = [col for col in transposed_data if not all(cell == '' for cell in col)]
        result = list(map(list, zip(*filtered_data)))
        return result

    def extract_text_and_tables(self, page: Dict):
        bottom = 0
        # FIXME
        # tables = page.find_tables(table_settings={'intersection_x_tolerance': 1})
        tables = page['tables']

        if len(tables) >= 1:
            count = len(tables)
            for table in tables:
                # FIXME
                # bbox = table.bbox
                # if bbox[3] < bottom:
                #     pass
                bbox = table['bbox']
                if bbox[3] < bottom:
                    pass
                else:
                    count -= 1
                    top = bbox[1]
                    text = self.check_lines(page, top, bottom)
                    text_list = text.split('\n')
                    for _t in range(len(text_list)):
                        # FIXME
                        # self.all_text[self.allrow] = {'page': page.page_number, 'allrow': self.allrow,
                        #                               'type': 'text', 'inside': text_list[_t]}
                        self.all_text[self.allrow] = {'page': page['page_no'], 'allrow': self.allrow,
                                                      'type': 'text', 'inside': text_list[_t]}
                        self.allrow += 1

                    buttom = bbox[3]
                    # FIXME
                    # new_table = table.extract()
                    new_table = table['rows']
                    r_count = 0
                    for r in range(len(new_table)):
                        row = new_table[r]
                        if row[0] is None:
                            r_count += 1
                            for c in range(len(row)):
                                if row[c] is not None and row[c] not in ['', ' ']:
                                    if new_table[r - r_count][c] is None:
                                        new_table[r - r_count][c] = row[c]
                                    else:
                                        new_table[r - r_count][c] += row[c]
                                    new_table[r][c] = None
                        else:
                            r_count = 0

                    end_table = []
                    for row in new_table:
                        if row[0] != None:
                            cell_list = []
                            cell_check = False
                            for cell in row:
                                if cell != None:
                                    cell = cell.replace('\n', '')
                                else:
                                    cell = ''
                                if cell != '':
                                    cell_check = True
                                cell_list.append(cell)
                            if cell_check == True:
                                end_table.append(cell_list)
                    end_table = self.drop_empty_cols(end_table)

                    for row in end_table:
                        # FIXME
                        # self.all_text[self.allrow] = {'page': page.page_number, 'allrow': self.allrow,
                        #                               'type': 'excel', 'inside': str(row)}
                        self.all_text[self.allrow] = {'page': page['page_no'], 'allrow': self.allrow,
                                                      'type': 'excel', 'inside': str(row)}
                        self.allrow += 1

                    if count == 0:
                        text = self.check_lines(page, '', buttom)
                        text_list = text.split('\n')
                        for _t in range(len(text_list)):
                            # FIXME
                            # self.all_text[self.allrow] = {'page': page.page_number, 'allrow': self.allrow,
                            #                               'type': 'text', 'inside': text_list[_t]}
                            self.all_text[self.allrow] = {'page': page['page_no'], 'allrow': self.allrow,
                                                          'type': 'text', 'inside': text_list[_t]}
                            self.allrow += 1

        else:
            text = self.check_lines(page, '', '')
            text_list = text.split('\n')
            for _t in range(len(text_list)):
                # FIXME
                # self.all_text[self.allrow] = {'page': page.page_number, 'allrow': self.allrow,
                #                               'type': 'text', 'inside': text_list[_t]}
                self.all_text[self.allrow] = {'page': page['page_no'], 'allrow': self.allrow,
                                              'type': 'text', 'inside': text_list[_t]}
                self.allrow += 1

        first_re = '[^计](?:报告(?:全文)?(?:（修订版）|（修订稿）|（更正后）)?)$'
        end_re = '^(?:\d|\\|\/|第|共|页|-|_| ){1,}'
        if self.last_num == 0:
            try:
                first_text = str(self.all_text[1]['inside'])
                end_text = str(self.all_text[len(self.all_text) - 1]['inside'])
                if re.search(first_re, first_text) and not '[' in end_text:
                    self.all_text[1]['type'] = '页眉'
                    if re.search(end_re, end_text) and not '[' in end_text:
                        self.all_text[len(self.all_text) - 1]['type'] = '页脚'
            except:
                # FIXME
                # print(page.page_number)
                print(page['page_no'])
        else:
            try:
                # first_text = str(self.all_text[self.last_num + 2]['inside'])
                # end_text = str(self.all_text[len(self.all_text) - 1]['inside'])
                # if re.search(first_re, first_text) and '[' not in end_text:
                #     self.all_text[self.last_num + 2]['type'] = '页眉'
                # if re.search(end_re, end_text) and '[' not in end_text:
                #     self.all_text[len(self.all_text) - 1]['type'] = '页脚'

                # 默认每页头两行为页眉
                self.all_text[self.last_num + 1]['type'] = '页眉'
                self.all_text[self.last_num + 2]['type'] = '页眉'
                # 默认每页最后一行为页脚
                self.all_text[len(self.all_text) - 1]['type'] = '页脚'

            except:
                # FIXME
                # print(page.page_number)
                print(page['page_no'])

        self.last_num = len(self.all_text) - 1


    def process_pdf(self):
        # FIXME
        #for i in range(len(self.pdf.pages)):
            # self.extract_text_and_tables(self.pdf.pages[i])
        for page in self.pages:
            self.extract_text_and_tables(page)

    def save_all_text(self, path):
        for key in self.all_text.keys():
            with open(path, 'a+', encoding='utf-8') as file:
                file.write(json.dumps(self.all_text[key], ensure_ascii=False) + '\n')

def process_all_pdfs_in_folder(folder_path):
    file_paths = glob.glob(f'{folder_path}/*')
    file_paths = sorted(file_paths, reverse=True)

    for file_path in file_paths:
        print(file_path)
        try:
            processor = PDFProcessor(file_path)
            processor.process_pdf()
            save_path = 'alltxt/' + file_path.split('/')[-1].replace('.pdf', '.txt')
            processor.save_all_text(save_path)
        except:
            print('check')


# folder_path = 'allpdf'
# process_all_pdfs_in_folder(folder_path)

def line_text_to_list(line_text):
    # line_text='["BeijingJinyu\'sBusinessPartern", \'借入资金-利息支出\', \'1,667,644.95\', \'1,146,268.90\']'
    # Cannot use json.loads() directly

    # print(f"{line_text=}")
    if line_text[:1] == '[' and line_text[-1:] == ']':
        items = line_text[1:-1].split("', '")
        items = [ item if item else "" for item in items ]
        if len(items) == 1:
            items[0] = items[0][1:-1]
        elif len(items) > 1:
            items[0] = items[0][1:]
            items[-1] = items[-1][:-1]

        # items = line_text[1:-1].split(', ')
        # items = [item.strip() for item in items]
        # items = [ item[1:-1] if item else "" for item in items ]
    else:
        items = json.loads(line_text)
    # print(f"{items=}\n")

    return items

def pdf2txt_file_to_text(pdf2txt_file: str, target_file_name: str):
    if not os.path.exists(target_file_name):

        all_lines = open(pdf2txt_file).readlines()
        all_lines = [ json.loads(line.strip()) for line in all_lines]
        all_lines = [ d for d in all_lines if 'page' in d]

        full_text = pdf2txt_lines_to_text(all_lines)

        with open(target_file_name, 'w') as fd:
            fd.write(full_text)


def pdf2txt_lines_to_text(all_lines):
    after_foot = True
    full_text = ""
    for line_data in all_lines:
        page_number = line_data['page']
        allrow = line_data['allrow']
        line_type = line_data['type']
        inside = line_data['inside'].strip()

        if line_type == '页脚':
            after_foot = True
            continue

        if line_type == "text":
            if inside == "" and after_foot:
                continue
            full_text += inside + "\n"
        elif line_type == "excel":
            # line = "| " + " | ".join(inside) + " |"
            line = "| " + " | ".join(line_text_to_list(inside)) + " |"
            full_text += line + "\n"
        elif line_type == '页眉':
            if page_number <= 1:
                full_text += inside + "\n"

        after_foot = False
            
    return full_text
