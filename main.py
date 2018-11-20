# -*- coding: utf-8 -*-
import os

import cv2
import re

a_img = cv2.imread('voe_images/02.jpg')
a_gray = cv2.cvtColor(a_img, cv2.COLOR_BGR2GRAY)

# AKAZEを使用して特徴量を抽出
akaze = cv2.AKAZE_create()
kp1, des1 = akaze.detectAndCompute(a_gray, None)

# Brute-force matcher生成
bf = cv2.BFMatcher()


def check_voe_images():
    min_num = 100
    nearest_image = ''
    dir_path = 'voe_images'
    for image_path in os.listdir(dir_path):
        if image_path.startswith('0'):
            b_img = cv2.imread('./%s/%s' % (dir_path, image_path))
            b_gray = cv2.cvtColor(b_img, cv2.COLOR_BGR2GRAY)
            kp2, des2 = akaze.detectAndCompute(b_gray, None)
            # 特徴量ベクトル同士をBrute-Force
            matches = bf.match(des1, des2)
            # 特徴点同士の距離でソート
            matches = sorted(matches, key=lambda x: x.distance)
            min_distance = matches[0].distance
            # 上位10個の合計距離とかで判断してもいいかも
            # matches[:10]
            if min_distance < min_num:
                min_num = min_distance
                nearest_image = image_path
    return min_num, nearest_image


print(check_voe_images())
