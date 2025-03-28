# coding=utf-8
# Created : 2023/1/9 16:33
# Author  : Zy
import os
import minio
import requests
from io import BytesIO
from common.utils import Util
from common.content_type_maps import suffix_content_type_maps


class OSS:
    def __init__(self, bucket_name):
        super(OSS, self).__init__()
        config = Util.get_config("ossApi.json").get("minio")
        minio_conf = {
            "access_key": config.get("access_key"),
            "secret_key": config.get("secret_key"),
            "endpoint": config.get("endpoint"),
            "secure": eval(config.get("secure"))
        }
        self.http = f'http://hz.gdhengdian.com/images/'
        self.client = minio.Minio(**minio_conf)
        self.bucket_name = bucket_name

    def upload(self, localfile: str):
        data_bytes = BytesIO(open(localfile, "rb").read())
        data_length = len(data_bytes.getvalue())
        suffix = Util.extract_file_suffix(localfile)
        content_type = suffix_content_type_maps.get(suffix, "")
        assert content_type, "unknow content-type"
        object_name = Util.generate_unique_file_name(suffix=suffix)
        self.client.put_object(
            bucket_name=self.bucket_name,
            object_name=object_name,
            data=data_bytes,
            length=data_length,
            content_type=content_type
        )
        data_bytes.close()
        return self.http + self.bucket_name + os.sep + object_name

    def download(self, remote_file):
        suffix = Util.extract_file_suffix(remote_file)
        req = requests.get(remote_file)
        local_file = Util.generate_temp_file_path(suffix=suffix)
        with open(local_file, "wb+") as fp:
            fp.write(req.content)
        return local_file


if __name__ == '__main__':
    oss = OSS("hz-images")
    url = oss.upload("/Users/zhangyu/Desktop/ml/_project_pytorch/口罩检测/test.png")
    print(url)
