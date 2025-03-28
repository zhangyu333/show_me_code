# coding=utf-8
# Created : 2023/1/10 09:08
# Author  : Zy
import os
import requests
from PIL import Image
from io import BytesIO
from common.utils import Util


def downloadImageFile(remote_url: str):
    img_b = BytesIO(requests.get(remote_url).content)
    pil_img = Image.open(img_b)
    local_path = Util.generate_temp_file_path(suffix=Util.extract_file_suffix(remote_url))
    pil_img.save(local_path, "PNG")
    img_b.close()
    return local_path


def downloadFile(remote_url: str):
    local_path = Util.generate_temp_file_path(suffix=Util.extract_file_suffix(remote_url))
    with open(local_path, "wb+") as fp:
        fp.write(requests.get(remote_url).content)
    return local_path


def clearCache(local_path: str):
    os.remove(local_path)


def osCall(cmd):
    os.system(cmd)


if __name__ == '__main__':
    pass
