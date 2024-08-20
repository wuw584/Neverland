import os
import sys
import json


def param(executable_path):
    current_path = os.path.dirname(os.path.abspath(executable_path)) #读取当前路径
    param_file_path = os.path.join(current_path, 'das_params.json') #参数文件在统一文件夹下，第一行是数据文件夹绝对路径，第二行保存图片的绝对路径
    with open(param_file_path, 'r') as file:
        data = json.load(file)
        return data