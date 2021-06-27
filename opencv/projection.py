#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Description:
@Date:2021/06/26 23:02:45
@Author:ttt
"""

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