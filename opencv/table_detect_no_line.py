#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Description:
@Date:2021/06/22 21:18:53
@Author:ttt
"""


import cv2
import numpy as np


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
    vertical_sum = np.sum(erode_image, axis=0)
    
    contours, hierarchy = cv2.findContours(erode_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    cells = []
    for k, cn in enumerate(contours[::-1]):
        x, y, w, h = cv2.boundingRect(cn)
        if h < 5 or w < 5:
            continue
        cell = np.array([x, y, x+w, y+h], dtype=np.int16)
        cells.append(cell)
    return cells, vertical_sum, erode_image


def is_inline(a, b, thr=0.15):
    hiou = min(b[3], a[3]) - max(b[1], a[1])
    flag = False
    if hiou > 0:
        h = min(b[3] - b[1], a[3] - a[1])
        flag = hiou/h > thr
    return flag 

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


def get_cut_point(vertical_sum, cells, s, e):
    cut_point = []
    for k in range(s, e+1):
        if k == s:
            j = cells[k][0]
            p = np.argmin(vertical_sum[0:j])
            i = cells[k][2]
        else:
            j = cells[k][0]
            if j < i:
                continue
            p = i + np.argmin(vertical_sum[i:j])
            i = cells[k][2]
        cut_point.append(p)
        
    return cut_point


def get_col(cell, cut_point):
    sx = cell[0]
    ex = cell[2]
    h = cell[3] - cell[1]
    i = 0
    while i < len(cut_point) and sx > cut_point[i]-h/2:
        i += 1
    si = i -1
    
    j = si
    while j < len(cut_point) and ex > cut_point[j]+h/2:
       j += 1
    ei = j-1
    return si, ei   



