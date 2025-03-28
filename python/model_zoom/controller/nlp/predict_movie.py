import json
from common.utils import Util
import joblib
import pandas as pd
from textrank4zh import TextRank4Keyword


app_path = Util.app_path()

def cut_keyword(text):
    tr4w = TextRank4Keyword()
    keyword_ls = []
    if type(text) != float:
        tr4w.analyze(text=text, window=2, lower=True)
        # print("关键词：")

        for item in tr4w.get_keywords(num=3, word_min_len=2):
            # print(item.word, item.weight)
            keyword_ls.append(item.word)
    return keyword_ls


def pridict(data):
    # --------------------ACTORS------------------------------

    data_dic = {}
    ACTORS = data["ACTORS"].split("/")

    with open(app_path + "/static/movie/ACTORS_map.json", "r", encoding="utf-8") as f:
        ACTORS_map = json.loads(f.read())

    num = 0
    ACTORS_rate = 0
    for i in ACTORS:
        if i in ACTORS_map.keys():
            num += 1
            ACTORS_rate += ACTORS_map[i]
    ACTORS_rate = round(ACTORS_rate / num, 3)
    data_dic["ACTORS"] = ACTORS_rate

    # --------------------DIRECTORS------------------------------

    DIRECTORS = data["DIRECTORS"].split("/")

    with open(app_path + "/static/movie/DIRECTORS_map.json", "r", encoding="utf-8") as f:
        DIRECTORS_map = json.loads(f.read())

    num = 0
    DIRECTORS_rate = 0
    for i in DIRECTORS:
        if i in DIRECTORS_map.keys():
            num += 1
            DIRECTORS_rate += DIRECTORS_map[i]
    DIRECTORS_rate = round(DIRECTORS_rate / num, 3)
    data_dic["DIRECTORS"] = DIRECTORS_rate

    # ----------------------GENRES--------------------------------

    GENRES = data["GENRES"].split("/")

    with open(app_path + "/static/movie/GENRES_map.json", "r", encoding="utf-8") as f:
        GENRES_map = json.loads(f.read())

    num = 0
    GENRES_rate = 0
    for i in GENRES:
        if i in GENRES_map.keys():
            num += 1
            GENRES_rate += GENRES_map[i]
    GENRES_rate = round(GENRES_rate / num, 3)
    data_dic["GENRES"] = GENRES_rate

    # -----------------------STORYLINE-----------------------------

    with open(app_path + "/static/movie/STORYLINE.txt", "r", encoding="utf-8") as f:
        story_list = f.read().split("/")

    nums = 1

    for i in cut_keyword(data["STORYLINE"]):
        if i not in story_list:
            data_dic[f"STORYLINE{nums}"] = len(story_list) - 1
        else:
            data_dic[f"STORYLINE{nums}"] = story_list.index(i)
        nums += 1

    # ----------------------------------------------------------

    df = pd.DataFrame(data_dic, index=[0])
    # print(df)

    clf = joblib.load(app_path + "/static/movie/train_model.m")  # 调用模型
    y_rf = clf.predict(df)
    return float(y_rf[0])


if __name__ == "__main__":
    # ### ['ACTORS', 'DIRECTORS', 'GENRES', 'STORYLINE']
    # ###  演员1/演员2    导演      类型1/类型2    简介

    data = {
        "ACTORS": "大鹏/李雪琴/尹正/王迅/王圣迪/马丽/宋茜/白宇/贾冰/杨迪/潘斌龙/倪虹洁/乔杉/于洋/刘金山/曹炳琨/梁超/张林/老四/李胤维/夏甄/邓飞/李妮妮/张婉儿/娄乃鸣/陈祉希/其那日图/兰西雅/方圆圆/林若惜/大能/吴昊宸",
        "DIRECTORS": "大鹏",
        "GENRES": "剧情/喜剧",
        "STORYLINE": "落魄中年魏平安（大鹏 饰）以直播带货卖墓地为生，他的客户韩露（宋茜 饰）过世后被造谣抹黑，魏平安路见不平，辟谣跑断腿，笑料频出，反转不断，而他自己也因此陷入到新的谣言和网暴之中。",
    }
    pridict(data)
