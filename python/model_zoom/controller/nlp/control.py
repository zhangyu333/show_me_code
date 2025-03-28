# coding=utf-8
# Created : 2023/1/9 17:51
# Author  : Zy
import pypinyin
import jieba.analyse
from controller.nlp.poem_gen import poemGen
from controller.nlp.story_gen import StoryGenerate
from controller.nlp.emo_analysis import getSimEmo
from controller.nlp.emo_analysis import emotionAnalysis
from controller.nlp import predict_movie

storygenerate = StoryGenerate()
emotionAnalysis = emotionAnalysis


def genPoem(target: str, type: str, fos: int, multi: bool, top: int):
    result = poemGen(target, type, fos, multi, top)
    return result


def genStory(title: str, content: str) -> str:
    story = storygenerate.generate(title, content)
    return story


def word2Pinyin(word: str):
    result = pypinyin.pinyin(word, heteronym=True)
    result = [r[0] for r in result]
    result = " ".join(result)
    return result


def getSimEmoInfo(sentence: str) -> dict:
    result = getSimEmo(sentence)
    return result


def extractTitle(sentence: str):
    title = jieba.analyse.extract_tags(sentence, topK=1)[0]
    return title


def movieScorePredict(data: dict) -> float:
    score = predict_movie.pridict(data)
    return score
