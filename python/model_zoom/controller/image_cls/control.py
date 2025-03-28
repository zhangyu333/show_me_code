# coding=utf-8
# Created : 2023/1/9 17:51
# Author  : Zy
from controller.image_cls.money_rec import MoneyRec
from controller.image_cls.vegetable_rec import VegetableRec
from controller.image_cls.zcy_rec import RecZCY


moneyrec = MoneyRec()
vegetablerec = VegetableRec()
rec_zcy = RecZCY()


def moneyRec(filename: str):
    res_result = moneyrec.infer(filename)
    return res_result


def vegetableRec(filename: str):
    res_result = vegetablerec.infer(filename)
    return res_result

def recZCY(filename: str):
    res_result = rec_zcy.main(filename)
    return res_result