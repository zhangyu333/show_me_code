# coding=utf-8
# Created : 2022/12/30 14:14
# Author  : Zy
from application import app
from controller.nlp.routes import nlp
from controller.voice.routes import voice
from controller.vision.routes import vision
from controller.detect.routes import obj_detect
from controller.image_cls.routes import image_cls
from controller.image_handle.routes import image_handle

app.register_blueprint(nlp, url_prefix='/nlp-v1')
app.register_blueprint(voice, url_prefix='/voice-v1')
app.register_blueprint(vision, url_prefix='/vision')
app.register_blueprint(image_cls, url_prefix='/image-cls-v1')
app.register_blueprint(obj_detect, url_prefix='/obj-detect-v1')
app.register_blueprint(image_handle, url_prefix='/image-handle-v1')
