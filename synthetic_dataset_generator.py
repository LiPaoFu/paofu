import cv2
import numpy as np
import os
import random

# 创建数据集目录
os.makedirs("dataset/images", exist_ok=True)
os.makedirs("dataset/masks", exist_ok=True)


def draw_wall(img, mask):
    """绘制外墙和内墙"""
    # 外墙
    cv2.rectangle(img, (10, 10), (118, 118), (0, 0, 0), 2)
    cv2.rectangle(mask, (10, 10), (118, 118), 1, 2)

    # 随机内墙
    if random.random() > 0.5:
        x = random.randint(30, 90)
        cv2.line(img, (x, 15), (x, 113), (0, 0, 0), 2)
        cv2.line(mask, (x, 15), (x, 113), 1, 2)


def add_door_window(img, mask):
    """在墙上添加门窗"""
    walls = [
        [(10, 60), (118, 60)],  # 水平墙
        [(60, 10), (60, 118)]  # 垂直墙
    ]

    for (x1, y1), (x2, y2) in walls:
        if random.random() > 0.7:  # 30%概率添加门
            pos = random.randint(20, 80)
            cv2.rectangle(img, (x1 + pos - 8, y1 - 2), (x1 + pos + 8, y1 + 2), (0, 255, 0), -1)
            cv2.rectangle(mask, (x1 + pos - 8, y1 - 2), (x1 + pos + 8, y1 + 2), 2, -1)

        if random.random() > 0.7:  # 30%概率添加窗
            pos = random.randint(20, 80)
            cv2.rectangle(img, (x1 + pos - 10, y1 - 4), (x1 + pos + 10, y1 + 4), (255, 0, 0), -1)
            cv2.rectangle(mask, (x1 + pos - 10, y1 - 4), (x1 + pos + 10, y1 + 4), 3, -1)


def add_furniture(img, mask):
    """添加家具"""
    # 床
    if random.random() > 0.5:
        x, y = random.randint(15, 80), random.randint(15, 80)
        cv2.rectangle(img, (x, y), (x + 40, y + 25), (0, 0, 255), -1) # 蓝色
        cv2.rectangle(mask, (x, y), (x + 40, y + 25), 4, -1)

    # 桌子
    if random.random() > 0.5:
        x, y = random.randint(15, 80), random.randint(15, 80)
        cv2.circle(img, (x, y), 8, (255, 0, 255), -1) # 粉色
        cv2.circle(mask, (x, y), 8, 5, -1)

    # 沙发
    if random.random() > 0.5:
        x, y = random.randint(15, 80), random.randint(15, 80)
        cv2.rectangle(img, (x, y), (x + 30, y + 10), (0, 255, 255), -1) # 洋红色
        cv2.rectangle(mask, (x, y), (x + 30, y + 10), 6, -1)


# 生成500张复杂户型图
for i in range(500):
    img = np.zeros((128, 128, 3), dtype=np.uint8) + 255
    mask = np.zeros((128, 128), dtype=np.uint8)

    draw_wall(img, mask)
    add_door_window(img, mask)
    add_furniture(img, mask)

    cv2.imwrite(f"dataset/images/{i}.png", img)
    cv2.imwrite(f"dataset/masks/{i}.png", mask)