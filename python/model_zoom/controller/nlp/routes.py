# coding=utf-8
# Created : 2023/1/9 17:51
# Author  : Zy
import json
from flask import request
from flask import Blueprint
from controller.nlp.control import *

nlp = Blueprint('nlp', __name__)


@nlp.route("/generate_poem", methods=["POST"])
def generatePoem():
    formdata = request.form
    top = formdata.get("top")
    fos = formdata.get("fos")
    type = formdata.get("type")
    multi = formdata.get("multi")
    target = formdata.get("target")
    multi = True if multi else False
    fos = int(fos)
    top = int(top)
    result = genPoem(target, type, fos, multi, top)
    return {
        "code": 200,
        "data": result,
        "message": ""
    }


@nlp.route("/generate_story", methods=["POST"])
def generateStory():
    formdata = request.form
    title = formdata.get("title")
    context = formdata.get("context")
    story = genStory(title, context)
    return {"data": {"story": story}, "code": 200, "message": ""}


@nlp.route("/word_to_pinyin", methods=["POST"])
def word2Pinyin_():
    formdata = request.form
    word = formdata.get("word")
    pinyin = word2Pinyin(word)
    return {"data": {"pinyin": pinyin}, "code": 200, "message": ""}


@nlp.route("/ds_semotion_analysis", methods=["POST"])
def emotionAnalysis_():
    formdata = request.form
    text = formdata.get("text")
    result = emotionAnalysis(text)
    return {"data": result, "code": 200, "message": ""}


@nlp.route("/get_sim_emo_info", methods=["POST"])
def getSimEmoInfo_():
    formdata = request.form
    text = formdata.get("text")
    result = getSimEmoInfo(text)
    return {"data": json.dumps(result), "code": 200, "message": ""}


@nlp.route("/extract-title", methods=["POST"])
def extractTitle_():
    formdata = request.form
    text = formdata.get("text")
    result = extractTitle(text)
    return {"data": {"title": result}, "code": 200, "message": ""}



@nlp.route("/movie-score-predict", methods=["POST"])
def movieScorePredict_():
    formdata = request.form
    data = formdata.get("data")
    data = json.loads(data)
    score = movieScorePredict(data)
    return {"data": {"score": score}, "code": 200, "message": ""}