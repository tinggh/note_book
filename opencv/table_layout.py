#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Description:
@Date:2021/06/23 19:27:42
@Author:ttt
"""

import numpy as np


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
    i = 0
    while i < len(cut_point) and sx > cut_point[i]:
        i += 1
    si = i -1
    
    j = si
    while j < len(cut_point) and ex > cut_point[j]:
       j += 1
    ei = j-1
    return si, ei    


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