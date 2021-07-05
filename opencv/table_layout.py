#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Description:
@Date:2021/06/23 19:27:42
@Author:ttt
"""


import cv2
import numpy as np
from numpy.lib.twodim_base import diag


def is_inline(a, b, thr=0.15):
    hiou = min(b[3], a[3]) - max(b[1], a[1])
    ha = a[3] - a[1]
    hb = b[3] - b[1]
    flag = False
    if hiou > 0:
        flag = hiou/min(hb, ha) > thr
    return flag 



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



def get_index(sx, ex, cut_point, threshold=0):
    if ex < cut_point[0]-threshold:
        ei = -1
        si = -1
        return si, ei
    
    if sx > cut_point[-1]+threshold:
        si = len(cut_point)
        ei = len(cut_point)
        return si, ei
    
    i = 0
    while i < len(cut_point) and sx > cut_point[i]-threshold:
        i += 1
    si = i -1
    
    j = si
    while j < len(cut_point) and ex > cut_point[j]+ threshold:
       j += 1
    ei = j-1
    return si, ei

    

def cell_layout(cells, x_cut_point):
    ocs = []
    r = 0
    frow = cells[0]
    for k, cell in enumerate(cells):
        sc, ec = get_index(cell[0], cell[2], x_cut_point, threshold=(cell[3] - cell[1])*0.3)
        if not is_inline(frow, cell):
            frow = cell
            r += 1
        dc = dict(
            box=cell.tolist(),
            start_row=r,
            end_row=r,
            start_col=sc,
            end_col=ec
        )
        ocs.append(dc)
    ocs = post_process(ocs)
    table = dict(cells=ocs)        
    return table


def xy2xyxy(box):
    xyxy =[
        [box[0], box[1]], 
        [box[2], box[1]],
        [box[2], box[3]], 
        [box[0], box[3]]]
    return xyxy


def merge(cells):
    points = []
    for c in cells:
        box = xy2xyxy(c["box"])
        points.extend(box)
    x, y, w, h = cv2.boundingRect(np.array(points))
    box = [x, y, w+x, h+y]
    return box
    
    
def post_process(cells):
    i = 0
    while i<len(cells)-1:
        cc = cells[i]
        j = i + 1
        while j < len(cells):
            nc = cells[j]
            if nc["start_row"] <= cc["end_row"]+1 and nc["start_col"] <= cc["end_col"] and is_inline(cc["box"], nc["box"]):
                j += 1
            else:
                break
        box = cc["box"] if j==i+1 else merge(cells[i:j])
        cc["box"] = box
        cc["end_col"] = cells[j-1]["end_col"]
        cc["end_row"] = cells[j-1]["end_row"]
        del cells[i+1:j]
        # i = max(i+1, j-1)
        i = i+1 if j==i+1 else i-1
            
    return cells



# lsd = cv2.createLineSegmentDetector(0, _scale=1)
ld = cv2.ximgproc.createFastLineDetector()
def get_lines(x_cut_point, y_cut_point, dilated_col, dilated_row):
    vlines = ld.detect(dilated_col)
    hlines = ld.detect(dilated_row)
    lines = []
    
    if vlines is not None:
        vlines = np.squeeze(vlines)
        vlines = sorted(vlines, key=lambda x: (x[0], x[1]))
        xmin, xmax = vlines[0][0], vlines[0][2]
        lines.append([vlines[0][0], vlines[0][1], vlines[0][2], vlines[0][3], "obvl"])
        for i in range(len(vlines)-1):
            if abs(vlines[i+1][0] - vlines[i][0]) < 5:
                continue
            else:
                line = [vlines[i+1][0], vlines[i+1][1], vlines[i+1][2], vlines[i+1][3], "obvl"]
                lines.append(line)
            
    if hlines is not None:
        hlines = np.squeeze(hlines)
        hlines = sorted(hlines, key=lambda x: (x[1], x[0]))
        ymin, ymax = hlines[0][1], hlines[0][3]
        lines.append([hlines[0][0], hlines[0][1], hlines[0][2], hlines[0][3], "obhl"])
        for i in range(len(hlines)-1):
            if abs(hlines[i+1][1] - hlines[i][1]) < 5:
                continue
            else:
                line = [hlines[i+1][0], hlines[i+1][1], hlines[i+1][2], hlines[i+1][3], "obhl"]
                lines.append(line)
            
    # for x in x_cut_point:
    #     is_unobvl = True
    #     for line in lines:
    #         if abs(x-line[0]) < 25 and lines[-1] == "obvl":
    #             is_unobvl = False
    #             break
    #     if is_unobvl:
    #         lines.append([x, y_cut_point[0], x, y_cut_point[-1], "unobvl"])
                
    # for y in y_cut_point:
    #     is_unobhl = True
    #     for line in lines:
    #         if abs(y-line[1]) < 10 and lines[-1] == "obhl":
    #             is_unobhl = False
    #             break
    #     if is_unobhl:
    #         lines.append([x_cut_point[0], y, x_cut_point[-1], y, "unobhl"])
    return lines