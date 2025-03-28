# coding=utf-8
# Created : 2023/1/9 15:38
# Author  : Zy
import random

class VisionUtil:


    def getColors(self, class_nums: int):
        colors = [
            [random.randint(0, 256), random.randint(0, 256), random.randint(0, 256)] for i in range(class_nums)
        ]
        return colors