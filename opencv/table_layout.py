#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Description:
@Date:2021/06/23 19:27:42
@Author:ttt
"""

def is_inline(a, b, thr=0.5):
    hiou = min(b[3], a[3]) - max(b[1], a[1])
    if hiou > 0:
        h = min(b[3] - b[1], a[3] - a[1])
        return h * 1.0 / h > thr
    else:
        return False


def find_max_cols(cells):
    max_cols = 1
    i = 0
    s = i
    e = i
    while i < len(cells):
        sc = cells[i]
        cols = 1
        j = i+1
        while j < len(cells):
            ec = cells[j]
            if is_inline(sc, ec):
                cols += 1
            else:
                break
        i = j
        if cols > max_cols:
            max_cols = cols
            s = i
            e = j-1
    return max_cols, s, e


def cell_layout(cells):
    