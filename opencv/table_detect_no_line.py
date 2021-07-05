#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Description:
@Date:2021/06/22 21:18:53
@Author:ttt
"""


import os
import os.path as P

import cv2
import numpy as np

from table_layout import cell_layout, quick_sort, is_inline, get_lines


def get_cells(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, -5)

    rows, cols = binary.shape
    scale = 20
    # 识别横线:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (cols // scale, 1))
    dilated_col = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    # 识别竖线：
    scale = 20
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, rows // scale))
    dilated_row = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    # # 将识别出来的横竖线合起来
    # bitwise_and = cv2.bitwise_and(dilated_col, dilated_row)
    # cv2.imwrite("tmp/excel_bitwise_and.jpg", bitwise_and)

    # 标识表格轮廓
    merge = cv2.add(dilated_col, dilated_row)
    new_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    merge = cv2.dilate(merge, new_kernel, iterations=2)

    # 两张图片进行减法运算，去掉表格框线
    merge2 = cv2.subtract(binary, merge)
    new_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 2))
    dilate_image = cv2.dilate(merge2, new_kernel, iterations=3)
    new_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    erode_image = cv2.erode(dilate_image, new_kernel, iterations=1)
    cv2.imwrite("tmp/erode_image.jpg", erode_image)
    vertical_sum = np.sum(erode_image, axis=0)
    
    contours, hierarchy = cv2.findContours(erode_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    cells = []
    for k, cn in enumerate(contours[::-1]):
        x, y, w, h = cv2.boundingRect(cn)
        if h < 5 or w < 5:
            continue
        cell = np.array([x, y, x+w, y+h], dtype=np.int16)
        cells.append(cell)
    cells = quick_sort(cells, 0, len(cells)-1)
    s, e = find_max_cols(cells)
    x_cut_point = get_x_cut_point(vertical_sum, cells, s, e)
    y_cut_point = get_y_cut_point(cells)
    return cells, x_cut_point, y_cut_point, dilated_col, dilated_row


def find_max_cols(cells):
    max_cols = 1
    i = 0
    s = e = i
    while i < len(cells):
        sc = cells[i]
        j = i+1
        while j < len(cells):
            ec = cells[j]
            if is_inline(sc, ec, thr=0):
                j += 1
            else:
                break
        
        if j-i > max_cols:
            max_cols = j-i
            s = i
            e = j-1
        i = j
    return s, e


def get_x_cut_point(vertical_sum, cells, s, e):
    cut_point = []
    
    for k in range(s, e+1):
        if k == s:
            # j = cells[k][0]
            # p = np.argmin(vertical_sum[0:j])
            p = min([c[0] for c in cells])
            i = cells[k][2]
        else:
            j = cells[k][0]
            if j < i:
                continue
            p = i + np.argmin(vertical_sum[i:j])
            i = cells[k][2]
        cut_point.append(p)
    p = max([c[2] for c in cells[:-10]])
    cut_point.append(p)
    return cut_point



def get_y_cut_point(cells):
    cut_point = []
    i = 0
    rows = []
    while i < len(cells)-1:
        sc = cells[i]
        j = i + 1
        while j < len(cells):
            ec = cells[j]
            if is_inline(sc, ec, thr=0):
                j += 1
            else:
                break
        rows.append([i, j])
        i = j
    # si, ei = rows[0]
    # cut_point.append(min([cells[j][1] for j in range(si, ei)])-5)
    for i in range(len(rows)):
        si, ei = rows[i]
        if ei-si > 2:
            break
    
    start_row = i
    si, ei = rows[start_row]
    print(i, si, ei)
    cut_point.append(min([cells[j][1] for j in range(si, ei)])-5)
    for i in range(start_row, len(rows)-1):
        si, ei = rows[i]
        nsi, nei = rows[i+1]
        py = (max([cells[j][3] for j in range(si, ei)]) + min([cells[j][1] for j in range(nsi, nei)])) / 2 
        cut_point.append(py)
    si, ei = rows[i+1]
    cut_point.append(max([cells[j][3] for j in range(si, ei)])+5)
    return cut_point


if __name__ == '__main__':
    import shutil
    from pdf_to_image import pyMuPDF_fitz_cvimg
    _base_dir = "/home/ting/Documents/data/finance/100家报表"
    imgs_dir = "/home/ting/Documents/data/finance/images/"
    label_dir = "/home/ting/Documents/data/finance/labels_auto/"
    if not os.path.exists(imgs_dir):
        os.makedirs(imgs_dir)
        os.makedirs(label_dir)
        
    count = 0
    # for root, dirs, files in os.walk(_base_dir):
    #     for d in dirs:
    #         for img_name in files:
    
    for r in os.listdir(_base_dir):
        for d in os.listdir(os.path.join(_base_dir, r)):
            for img_name in os.listdir(os.path.join(_base_dir,r, d)):
                if img_name.startswith('.'):
                    continue
                file_path = P.join(_base_dir, r, d, img_name)
                imgs = pyMuPDF_fitz_cvimg(file_path)
                for image in imgs:
                    cv2.imwrite(imgs_dir + str(count) + '.jpg', image)
                    # cells, x_cut_point, y_cut_point, dilated_col, dilated_row = get_cells(image)
                    # lines = get_lines(x_cut_point, y_cut_point, dilated_col, dilated_row)
                    # table = cell_layout(cells, x_cut_point)
                    # cells = table["cells"]
                    # for k, oc in enumerate(cells):
                    #     sr = oc["start_row"]
                    #     er = oc["end_row"]
                    #     sc = oc["start_col"]
                    #     ec = oc["end_col"]
                    #     x1, y1, x2, y2 = oc["box"]
                    #     cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255))
                    #     cv2.putText(image, f'{k}_{sr}:{er}_{sc}:{ec}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                        
                        
                    # for l in lines:
                    #     if l[-1] == 'obhl':
                    #         cv2.line(image, (int(l[0]), int(l[1])), (int(l[2]), int(l[3])), (255, 0, 0), 1, cv2.LINE_AA)
                    #     elif l[-1] == 'obvl':
                    #         cv2.line(image, (int(l[0]), int(l[1])), (int(l[2]), int(l[3])), (0, 255, ), 1, cv2.LINE_AA)
                    #     elif l[-1] == 'unobhl':
                    #         cv2.line(image, (int(l[0]), int(l[1])), (int(l[2]), int(l[3])), (0, 0, 255), 1, cv2.LINE_AA)
                    #     elif l[-1] == 'unobvl':
                    #         cv2.line(image, (int(l[0]), int(l[1])), (int(l[2]), int(l[3])), (0, 0, 255), 1, cv2.LINE_AA)
                    # label_file = label_dir + str(count) + '.txt'  
                    # with open(label_file, 'a+', encoding="utf-8") as f:
                    #     f.write("\n".join([",".join([str(l[0]), str(l[1]), str(l[2]), str(l[3]), l[4]]) for l in lines]))
                        
                    # cv2.imwrite(f"tmp/result_{count}.jpg", image)
                    count += 1
