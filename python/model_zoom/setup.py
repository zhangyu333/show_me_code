# coding=utf-8
# Created : 2022/12/30 17:20
# Author  : Zy

from setuptools import setup, find_packages
setup(
    name='My Application',
    version='1.0',
    long_description=__doc__,
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=['Flask']
)