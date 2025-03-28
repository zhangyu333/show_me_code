# coding=utf-8
# Created : 2023/1/9 14:51
# Author  : Zy
import os
import uuid
import json
import datetime


class MyResp():
    def __call__(self):
        return {
            "message": "",
            "data": {},
            "code": 200
        }


class Util:

    @staticmethod
    def app_path():
        _file_path = os.path.dirname(__file__)
        return os.path.abspath(_file_path + "/../")

    @staticmethod
    def run_path():
        return os.getcwd()

    @staticmethod
    def unique_id():
        return str(uuid.uuid1()).replace('-', '')

    @staticmethod
    def current_hours():
        return datetime.datetime.now().strftime("%Y%m%d%H")

    @staticmethod
    def get_config(file_name):
        config = Util.app_path() + os.sep + "config" + os.sep + file_name
        with open(config, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config

    @staticmethod
    def cache_path():
        config = Util.get_config('common.json')
        return Util.app_path() + config["cache_path"]

    @staticmethod
    def extract_file_name(path):
        """
        根据路径提取文件名
        """
        return os.path.basename(path)

    @staticmethod
    def extract_file_suffix(path):
        """
        提取文件后缀
        """
        file_name = Util.extract_file_name(path)
        return file_name.split(".")[-1]

    @staticmethod
    def generate_unique_file_name(suffix=None, file_path=None):
        """
        根据文件后缀或者文件路径，生成唯一的文件名
        """
        if suffix:
            return Util.unique_id() + "." + suffix
        elif file_path:
            suffix = Util.extract_file_suffix(file_path)
            return Util.generate_unique_file_name(suffix=suffix)

    @staticmethod
    def generate_temp_file_path(suffix=None, file_name=None, file_path=None):
        """
        根据后缀生成一个临时文件路径
        """
        cache_path = Util.cache_path()
        os.makedirs(cache_path, exist_ok=True)
        if suffix:
            return cache_path + os.sep + Util.generate_unique_file_name(suffix=suffix)
        elif file_name:
            return cache_path + os.sep + file_name
        elif file_path:
            suffix = Util.extract_file_suffix(file_path)
            return Util.generate_temp_file_path(suffix=suffix)


if __name__ == '__main__':
    pass
