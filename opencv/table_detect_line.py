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
from table_layout import quick_sort, get_index, post_process


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


    # 将识别出来的横竖线合起来
    bitwise_and = cv2.bitwise_and(dilated_col, dilated_row)
    # 将焦点标识取出来
    ys, xs = np.where(bitwise_and > 0)
    # 横纵坐标数组
    y_cut_point = []
    x_cut_point = []
    # 通过排序，排除掉相近的像素点，只取相近值的最后一点
    # 这个10就是两个像素点的距离，不是固定的，根据不同的图片会有调整，基本上为单元格表格的高度（y坐标跳变）和长度（x坐标跳变）
    i = 0
    sort_x_point = np.sort(xs)
    for i in range(len(sort_x_point) - 1):
        if sort_x_point[i + 1] - sort_x_point[i] > 10:
            x_cut_point.append(sort_x_point[i])
        i = i + 1
    x_cut_point.append(sort_x_point[i])
    
    i = 0
    sort_y_point = np.sort(ys)
    for i in range(len(sort_y_point) - 1):
        if (sort_y_point[i + 1] - sort_y_point[i] > 10):
            y_cut_point.append(sort_y_point[i])
        i = i + 1
    y_cut_point.append(sort_y_point[i])     
            
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

    contours, hierarchy = cv2.findContours(erode_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    cells = []
    for k, cn in enumerate(contours[::-1]):
        x, y, w, h = cv2.boundingRect(cn)
        if h < 5 or w < 5:
            continue
        box = np.array([x, y, x+w, y+h], dtype=np.int16)
        cells.append(box)
    cells = quick_sort(cells, 0, len(cells)-1)
    return cells, x_cut_point, y_cut_point, erode_image
        

def line_cell_layout(cells, x_cut_point, y_cut_point):
    ocs = []
    for k, c in enumerate(cells):
        h = c[3] - c[1]
        sr, er = get_index(c[1], c[3], y_cut_point)
        sc, ec = get_index(c[0], c[2], x_cut_point, threshold=h)
        dc = dict(
            box=c.tolist(),
            start_row=sr,
            end_row=er,
            start_col=sc,
            end_col=ec
        )
        ocs.append(dc)
    ocs = post_process(ocs)
    table = dict(cells=ocs)
    return table


if __name__ == '__main__':
    _base_dir = "images/lines/"
    img_list = sorted(os.listdir(_base_dir))
    
    for i, img_name in enumerate(img_list):
        img_path = P.join(_base_dir, img_name)
        image = cv2.imread(img_path)
        cells, x_cut_point, y_cut_point, _ = get_cells(image)
        table = line_cell_layout(cells, x_cut_point, y_cut_point)
        cells = table["cells"]
        for k, oc in enumerate(cells):
            sr = oc["start_row"]
            er = oc["end_row"]
            sc = oc["start_col"]
            ec = oc["end_col"]
            x1, y1, x2, y2 = oc["box"]
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255))
            cv2.putText(image, f'{k}_{sr}:{er}_{sc}:{ec}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        cv2.imwrite(f"tmp/result_{i}.jpg", image)

        


