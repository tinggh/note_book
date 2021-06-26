#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Description:
@Date:2021/06/23 19:27:42
@Author:ttt
"""

import os
import os.path as P
import cv2

import numpy as np

from table_detect_no_line import (find_max_cols, get_cells, get_col, is_inline,
                                  get_cut_point)



def is_prior(a, b):
    flag = False
    if is_inline(a, b):
        flag = a[0] < b[0]
    else:
        flag = a[1] < b[1]
    return flag


def quick_sort(data_list, low, high):
    i = low
    j = high
    if i >= j:
        return data_list
 
    key = data_list[i]
    while i < j:
        while i < j and is_prior(key, data_list[j]):
            j = j - 1
        data_list[i] = data_list[j]
 
        while i < j and not is_prior(key, data_list[i]):
            i = i + 1
        data_list[j] = data_list[i]
 
    data_list[i] = key
 
    quick_sort(data_list, low, i - 1)
    quick_sort(data_list, j + 1, high)
    return data_list
    

def cell_layout(cells, vertical_sum):
    cells = quick_sort(cells, 0, len(cells)-1)
    s, e = find_max_cols(cells)
    cut_point = get_cut_point(vertical_sum, cells, s, e)
    
    oc = []
    r = 0
    for k, cell in enumerate(cells):
        sc, ec = get_col(cell, cut_point)
        if sc == 0:
            frow = cell
            r += 1
        dc = dict(
            box=cell.tolist(),
            row=r,
            start_col=sc,
            end_col=ec
        )
        oc.append(dc)
    table = dict(cells=oc)        
    return table



if __name__ == '__main__':
    _base_dir = "images"
    img_list = sorted(os.listdir(_base_dir))
    
    for i, img_name in enumerate(img_list):
        img_path = P.join(_base_dir, img_name)
        image = cv2.imread(img_path)
        cells, vertical_sum, _ = get_cells(image)
        table = cell_layout(cells, vertical_sum)
        cells = table["cells"]
        for k, oc in enumerate(cells):
            r = oc["row"]
            sc = oc["start_col"]
            ec = oc["end_col"]
            x1, y1, x2, y2 = oc["box"]
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255))
            cv2.putText(image, f'{r}_{sc}:{ec}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        cv2.imwrite(f"tmp/result_{i}.jpg", image)
