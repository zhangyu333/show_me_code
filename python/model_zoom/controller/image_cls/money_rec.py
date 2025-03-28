# -*- coding: utf-8 -*-
# @Time    : 2022/12/5 16:44
# @Author  : ZhangY
# @File    : money_rec
# @Project : PycharmProjects
import cv2
import random
import numpy as np
import onnxruntime as ort
from common.utils import Util


class MoneyRec:
    def __init__(self):
        super(MoneyRec, self).__init__()
        self.countries = ['日本', '澳大利亚', '中华人民共和国香港特别行政区', '韩国', '中华人民共和国']
        self.values = ['1000', '5000', '1', '10000', '100', '50', '5', '20', '500', '10', '2']
        self.years = ['1980-1996', '2018', '1984-1993', '1974-1994',
                      '2003', '1953-1958', '2018-2021', '1999',
                      '1992-1999', '2006-2009', '1983']
        self.model = ort.InferenceSession(Util.app_path() + "/models/cls/money.onnx",
                                          providers=["CPUExecutionProvider"])

    def infer(self, filename):
        img = cv2.imread(filename)
        img = cv2.resize(img, (112, 112), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None]
        x = np.ascontiguousarray(img).astype("float32")
        outputs = self.model.run(None, {"input": x})
        values_idx = np.argmax(outputs[0], 1)
        countries_idx = np.argmax(outputs[1], 1)
        years_idx = np.argmax(outputs[2], 1)

        value_score = outputs[0][0][values_idx]
        country_score = outputs[1][0][countries_idx]
        year_score = outputs[2][0][years_idx]

        result = {
            "面值": self.values[int(values_idx)],
            "国家": self.countries[int(countries_idx)],
            "年份": self.years[int(years_idx)],
            "面值置信度": float(value_score),
            "国家置信度": float(country_score),
            "年份置信度": float(year_score)
        }
        return result


if __name__ == '__main__':
    pass
