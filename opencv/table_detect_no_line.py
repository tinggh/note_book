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

_base_dir = "images"
img_list = sorted(os.listdir(_base_dir))

img_path = P.join(_base_dir, img_list[4])
raw = cv2.imread(img_path)
# h, w = raw.shape[:2]
# long_size = 1600
# scale = min(1, long_size / max(h, w))
# raw = cv2.resize(raw, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)


gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
binary = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, -5)
# 展示图片
cv2.imwrite("tmp/binary_picture.jpg", binary)


rows, cols = binary.shape
scale = 20
# 自适应获取核值
# 识别横线:
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (cols // scale, 1))
dilated_col = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
cv2.imwrite("tmp/excel_horizontal_line.jpg", dilated_col)


# 识别竖线：
scale = 20
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, rows // scale))
dilated_row = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
cv2.imwrite("tmp/excel_vertical_line.jpg", dilated_row)


# # 将识别出来的横竖线合起来
# bitwise_and = cv2.bitwise_and(dilated_col, dilated_row)
# cv2.imwrite("tmp/excel_bitwise_and.jpg", bitwise_and)


# 标识表格轮廓
merge = cv2.add(dilated_col, dilated_row)
cv2.imwrite("tmp/add_dilated_row_col.jpg", merge)
new_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
merge = cv2.dilate(merge, new_kernel, iterations=2)
cv2.imwrite("tmp/entire_excel_contour.jpg", merge)


# 两张图片进行减法运算，去掉表格框线
merge2 = cv2.subtract(binary, merge)
cv2.imwrite("tmp/binary_sub_excel_rect.jpg", merge2)



new_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 2))
dilate_image = cv2.dilate(merge2, new_kernel, iterations=3)
new_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
erode_image = cv2.erode(dilate_image, new_kernel, iterations=1)
cv2.imwrite('tmp/erode_image2.jpg', erode_image)


contours, hierarchy = cv2.findContours(erode_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(raw.copy(), contours, -1, (0, 0, 255), 3)
cv2.imwrite('tmp/contour_image.jpg', raw)    

cells = []
for k, cn in enumerate(contours[::-1]):
    x, y, w, h = cv2.boundingRect(cn)
    if h < 5 or w < 5:
        continue
    cell = np.array([x, y, x+w, y+h], dtype=np.int16)
    cells.append(cell)
    
    
from table_layout import cell_layout   
vertical_sum = np.sum(erode_image, axis=0)
table = cell_layout(cells, vertical_sum)
cells = table["cells"]
for k, oc in enumerate(cells):
    r = oc["row"]
    sc = oc["start_col"]
    ec = oc["end_col"]
    x1, y1, x2, y2 = oc["box"]
    cv2.rectangle(raw, (x1, y1), (x2, y2), (0, 0, 255))
    cv2.putText(raw, f'{r}_{sc}:{ec}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

cv2.imwrite("result.jpg", raw)





import matplotlib.pyplot as plt
import imutils
plt.figure("erode_image")
plt_erode_image = imutils.opencv2matplotlib(erode_image)
plt.imshow(plt_erode_image)
plt.savefig('erode_image.jpg')

plt.figure("projection")
vertical_sum = np.sum(erode_image, axis=0)
plt.plot(range(vertical_sum.shape[0]), vertical_sum)
plt.savefig('vertical_projec.jpg')


# data = np.float32(vertical_sum)
# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
# ret,label,center=cv2.kmeans(data,8,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)


# res = center[label.flatten()]
# plt.plot(range(res.shape[0]), res)
# plt.show()

vertical_sum = np.sum(erode_image, axis=0)
