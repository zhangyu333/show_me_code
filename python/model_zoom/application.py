# coding=utf-8
# Created : 2022/12/30 14:00
# Author  : Zy
import os
import json
from flask_cors import CORS
from flask import Flask, render_template, make_response

app = Flask(__name__)
app.config.from_pyfile(f'config{os.sep}base_setting.py')
CORS(app, supports_credentials=True, resources={r'/*': {"origins": "*"}})


@app.route("/", methods=["GET"])
def hello():
    return render_template('hello.html')


@app.errorhandler(404)
def not_found_error(error):
    result = {'status': 404, 'info': "Page not found"}
    response = make_response(json.dumps(result))
    return response


@app.errorhandler(500)
def internal_error(error):
    result = {'status': 500, 'info': f"Server exception: << {error} >>"}
    print(f"ERROR ERROR ERROR ERROR ERROR ERROR ===>>\n\tINFO:{result}")
    response = make_response(json.dumps(result))
    return response


@app.errorhandler(Exception)
def handle_exception(e):
    return internal_error(e)
