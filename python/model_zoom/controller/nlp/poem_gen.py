# -*- coding: utf-8 -*-
# @Time    : 2022/11/22 16:15
# @Author  : ZhangY
# @File    : generate
# @Project : PycharmProjects
import json
import glob
import random
import pypinyin
from common.utils import Util


def getRhythm(s1, s2):
    if s1 in ['qi', 'ti', 'yi', 'pi', 'di', 'ji', 'li', 'bi', 'ni', 'mi', 'xi', 'ri', 'si', 'zi', 'ci', 'shi', 'zhi',
              'chi'] and s2 in ['qi', 'ti', 'yi', 'pi', 'di', 'ji', 'li', 'bi', 'ni', 'mi', 'xi', 'ri', 'si', 'zi',
                                'ci', 'shi', 'zhi', 'chi']:
        if s1 in ['qi', 'ti', 'yi', 'pi', 'di', 'ji', 'li', 'bi', 'ni', 'mi', 'xi'] and s2 in ['qi', 'ti', 'yi', 'pi',
                                                                                               'di', 'ji', 'li', 'bi',
                                                                                               'ni', 'mi', 'xi']:
            return True
        if s1 in ['ri', 'si', 'zi', 'ci', 'shi', 'zhi', 'chi'] and s2 in ['ri', 'si', 'zi', 'ci', 'shi', 'zhi', 'chi']:
            return True
        return False
    if s1 in ['wu', 'ru', 'tu', 'pu', 'su', 'du', 'fu', 'gu', 'hu', 'ku', 'lu', 'zu', 'cu', 'bu', 'nu', 'mu', 'zhu',
              'chu', 'shu', 'qu', 'yu', 'ju', 'xu'] and s2 in ['wu', 'ru', 'tu', 'pu', 'su', 'du', 'fu', 'gu', 'hu',
                                                               'ku', 'lu', 'zu', 'cu', 'bu', 'nu', 'mu', 'zhu', 'chu',
                                                               'shu', 'qu', 'yu', 'ju', 'xu']:
        if s1 in ['qu', 'yu', 'ju', 'xu'] and s2 in ['qu', 'yu', 'ju', 'xu']:
            return True
        if s1 in ['wu', 'ru', 'tu', 'pu', 'su', 'du', 'fu', 'gu', 'hu', 'ku', 'lu', 'zu', 'cu', 'bu', 'nu', 'mu', 'zhu',
                  'chu', 'shu'] and s2 in ['wu', 'ru', 'tu', 'pu', 'su', 'du', 'fu', 'gu', 'hu', 'ku', 'lu', 'zu', 'cu',
                                           'bu', 'nu', 'mu', 'zhu', 'chu', 'shu']:
            return True
        return False
    if s1 in ['quan', 'juan', 'xuan', 'guan', 'ruan', 'tuan', 'suan', 'duan', 'huan', 'luan', 'zuan', 'cuan', 'nuan',
              'chuan', 'zhuan', 'shuan'] and s2 in ['quan', 'juan', 'xuan', 'guan', 'ruan', 'tuan', 'suan', 'duan',
                                                    'huan', 'luan', 'zuan', 'cuan', 'nuan', 'chuan', 'zhuan', 'shuan']:
        if s1 in ['quan', 'juan', 'xuan'] and s2 in ['quan', 'juan', 'xuan']:
            return True
        if s1 in ['guan', 'ruan', 'tuan', 'suan', 'duan', 'huan', 'luan', 'zuan', 'cuan', 'nuan', 'chuan', 'zhuan',
                  'shuan'] and s2 in ['guan', 'ruan', 'tuan', 'suan', 'duan', 'huan', 'luan', 'zuan', 'cuan', 'nuan',
                                      'chuan', 'zhuan', 'shuan']:
            return True
        return False
    if s1 in ["yun", "yuan", "me"] or s2 in ["yun", "yuan", "me"]:
        if s1 == s2:
            return True
        else:
            return False
    if s1 == "yan":
        s1 = "yian"
    if s2 == "yan":
        s2 = "yian"
    if s1 == "ye":
        s1 = "yie"
    if s2 == "ye":
        s2 = "yie"
    if s1 == "feng":
        s1 = "fong"
    if s2 == "feng":
        s2 = "fong"
    if s1 == "meng":
        s1 = "mong"
    if s2 == "meng":
        s2 = "mong"
    y = ["a", "o", "e", "i", "u"]
    e = ["ai", "ei", "ao", "ou", "an", "en", "ia", "ie", "in", "ue", "un", "ui", "iu"]
    s = ["ong", "ang", "eng", "uan", "ian"]
    if s1[-3:] == s2[-3:] and s1[-3:] in s:
        return True
    if s1[-2:] == s2[-2:] and s1[-2:] in e and s1[-3:] not in s and s2[-3:] not in s:
        return True
    if s1[-1:] == s2[-1:] and s1[-1:] in y and s1[-2:] not in e and s2[-2:] not in e:
        return True
    return False


def tongdiao(s1, s2):
    a = ["āīēōūǖ", "áíéóúǘ", "ǎǐěǒǔǚ", "àìèòùǜ"]
    for i in s1:
        if i not in "qwertyuiopasdfghjklzxcvbnm":
            ss1 = i
            break
        ss1 = "a"
    for i in s2:
        if i not in "qwertyuiopasdfghjklzxcvbnm":
            ss2 = i
            break
        ss2 = "a"
    for i in a:
        if ss1 in i and ss2 in i:
            return True
        if ss1 == ss2:
            return True
    return False


def readPoem(fos, type):
    poet = []
    for filename in glob.glob(Util.app_path() + f"/static/poet/*{type}*"):
        with open(filename, "r", encoding="utf-8") as fin:
            s = fin.read()
            temp = json.loads(s)
            for j in temp:
                for m in j["paragraphs"]:
                    if len(m) == 12 and fos == 5:
                        poet.append(m[0:5])
                        poet.append(m[6:11])
                    elif len(m) == 16 and fos == 7:
                        poet.append(m[0:7])
                        poet.append(m[8:15])
    return poet


def searchPoem(poet, f, target, multi, top):
    result = []
    matched = False
    for m in f:
        ans = [m]
        last_pinyin = pypinyin.lazy_pinyin(m[-1])[0]
        last_char = [m[-1]]
        for i in range(1, len(target)):
            for j in poet:
                if j[0] == target[i] and getRhythm(last_pinyin, pypinyin.lazy_pinyin(j[-1])[0]) and j not in ans and j[
                    -1] not in last_char and not tongdiao(pypinyin.pinyin(ans[-1][-1])[0][0],
                                                          pypinyin.pinyin(j[-1])[0][0]):
                    ans.append(j)
                    last_char.append(j[-1])
                    break
        if len(ans) == len(target):
            matched = True
            result.append(ans)
            if multi == False:
                break
            top -= 1
            if top == 0:
                break
    return result, matched


def poemGen(target, type="tang", fos=5, multi=False, top=5):
    ret = {"result": [], "info": "成功"}
    f = []
    matched = False
    can = True
    wrong_char = ""
    fin = open(Util.app_path() + f"/static/data{fos}.txt", "r", encoding="utf-8")
    t = fin.read()
    fin.close()
    for i in target:
        if i not in t:
            can = False
            wrong_char = i
            break
    if can:
        poet = readPoem(fos, type)
        random.shuffle(poet)
        for i in poet:
            if i[0] == target[0]:
                f.append(i)
                matched = True
        if matched:
            result, matched = searchPoem(poet, f, target, multi, top)
            if matched:
                ret["result"] = result
            else:
                ret["info"] = "没有找到符合" + target + "的诗句组合"
        else:
            ret["info"] = "没有找到以" + target[0] + "字开头的诗句"
    else:
        ret["info"] = "没有找到以" + wrong_char + "字开头的诗句"

    return ret


def getParse():
    import argparse
    parser = argparse.ArgumentParser(description="诗歌生成器")
    parser.add_argument('--category', choices=["tang", "song", "yuanqu"], default="tang", help="诗歌类型")
    parser.add_argument('--fos', choices=[5, 7], default=5, help="五言还是七言")
    parser.add_argument('--target', default="羊城风光", type=str, help="输入要生成的诗歌")
    parser.add_argument('--multi', type=bool, default=False, help="要开启多重结果检索")
    parser.add_argument('--top', type=int, default=5, help="多重结果输出最大多少首诗")
    args = parser.parse_args()
    return args
