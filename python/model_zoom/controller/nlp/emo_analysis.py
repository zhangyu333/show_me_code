# coding=utf-8
# Created : 2023/1/13 10:04
# Author  : Zy
import synonyms
import jieba.analyse
from snownlp import SnowNLP
from common.utils import Util


def delStopwords(words: list) -> list:
    with open(Util.app_path() + "/static/stopwords.txt", encoding="utf-8") as fp:
        stopwords = [w.strip() for w in fp.readlines()]
    tegether = words and stopwords
    words = [word for word in words if word not in tegether]
    return words


def emotionAnalysis(text: str) -> dict:
    analysis_result = SnowNLP(text)
    score = analysis_result.sentiments
    words = analysis_result.words
    words = delStopwords(words)
    emo = "积极评论" if score >= 0.5 else "消极评论"
    return {"emotion": emo, "word": ' '.join(words)}


def getSimEmo(s):
    results = {
        "nearby_scores": [],
        "nearby_words": [],
        "nearby_emo": [],
        "info": "success"
    }
    keywords = jieba.analyse.textrank(s, topK=1)
    keyword = keywords[0] if keywords else ""
    if not keyword:
        results["info"] = "extract keyword error, please check you sentence length"
        return results
    nearby_words, nearby_scores = synonyms.nearby(keyword, 6)
    results['nearby_scores'] = nearby_scores[1:]
    results['nearby_words'] = nearby_words[1:]
    for word in nearby_words[1:]:
        analysis_result = SnowNLP(word)
        emo_socre = analysis_result.sentiments
        results['nearby_emo'].append("积极" if emo_socre >= 0.5 else "消极")
    return results
