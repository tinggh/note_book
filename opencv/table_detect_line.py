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

_base_dir = "/home/ting/Documents/data/finance/选取的报表"
img_list = sorted(os.listdir(_base_dir))

img_path = P.join(_base_dir, img_list[0])
raw = cv2.imread(img_path)




gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
binary = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, -5)
# 展示图片
cv2.imwrite("tmp/binary_picture.jpg", binary)


rows, cols = binary.shape
scale = 40
# 自适应获取核值
# 识别横线:
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (cols // scale, 1))
eroded = cv2.erode(binary, kernel, iterations=1)
dilated_col = cv2.dilate(eroded, kernel, iterations=1)
cv2.imwrite("tmp/excel_horizontal_line.jpg", dilated_col)


# 识别竖线：
scale = 20
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, rows // scale))
eroded = cv2.erode(binary, kernel, iterations=1)
dilated_row = cv2.dilate(eroded, kernel, iterations=1)
cv2.imwrite("tmp/excel_vertical_line.jpg", dilated_row)


# 将识别出来的横竖线合起来
bitwise_and = cv2.bitwise_and(dilated_col, dilated_row)
cv2.imwrite("tmp/excel_bitwise_and.jpg", bitwise_and)


# 标识表格轮廓
merge = cv2.add(dilated_col, dilated_row)
cv2.imwrite("tmp/entire_excel_contour.jpg", merge)

# 两张图片进行减法运算，去掉表格框线
merge2 = cv2.subtract(binary, merge)
cv2.imwrite("tmp/binary_sub_excel_rect.jpg", merge2)



new_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
erode_image = cv2.morphologyEx(merge2, cv2.MORPH_OPEN, new_kernel)
cv2.imwrite('tmp/erode_image2.jpg', erode_image)


merge3 = cv2.add(erode_image, bitwise_and)
cv2.imwrite('tmp/merge3.jpg', merge3)


# 将焦点标识取出来
ys, xs = np.where(bitwise_and > 0)


# 横纵坐标数组
y_point_arr = []
x_point_arr = []
# 通过排序，排除掉相近的像素点，只取相近值的最后一点
# 这个10就是两个像素点的距离，不是固定的，根据不同的图片会有调整，基本上为单元格表格的高度（y坐标跳变）和长度（x坐标跳变）
i = 0
sort_x_point = np.sort(xs)
for i in range(len(sort_x_point) - 1):
    if sort_x_point[i + 1] - sort_x_point[i] > 10:
        x_point_arr.append(sort_x_point[i])
    i = i + 1
# 要将最后一个点加入
x_point_arr.append(sort_x_point[i])

i = 0
sort_y_point = np.sort(ys)
for i in range(len(sort_y_point) - 1):
    if (sort_y_point[i + 1] - sort_y_point[i] > 10):
        y_point_arr.append(sort_y_point[i])
    i = i + 1
y_point_arr.append(sort_y_point[i])



table_cell = []
for j in range(len(y_point_arr) -1):
    y1 = y_point_arr[j]
    y2 = y_point_arr[j+1]
    for i in range(len(x_point_arr) -1):
        x1 = x_point_arr[i]
        x2 = x_point_arr[i+1]
        box = np.array([x1, y1, x2, y2])
        table_cell.append(box)



for k, box in enumerate(table_cell):
    cv2.rectangle(raw, (box[0], box[1]), (box[2], box[3]), (0, 0, 255))
    cv2.putText(raw, str(k), (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

cv2.imwrite("result.jpg", raw)
