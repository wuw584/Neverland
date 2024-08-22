import os
import json


##get params
# param = gp.param(sys.argv[0])
# data_dir = param["data_dir"]
# das_read_clean_save_dir = param["das_read_clean_save_dir"]

# event_catalog_file = param["event_catalog_file"]
# event_cut_npy_save_dir = param["event_cut_npy_save_dir"]
# event_cut_plt_save_dir = param["event_cut_plot_save_dir" ]
# event_cut_file_name_prefix = param["event_cut_file_name_prefix"]
# dt_after = param["dt_after"]
# dt_before  =param["dt_before"]

# 创建文件夹
# if not os.path.exists(das_read_clean_save_dir):
#     os.makedirs(das_read_clean_save_dir)	# 创建文件夹


def param(executable_path):
    current_path = os.path.dirname(os.path.abspath(executable_path)) #读取当前路径
    param_file_path = os.path.join(current_path, 'das_params.json') #参数文件在统一文件夹下，第一行是数据文件夹绝对路径，第二行保存图片的绝对路径
    with open(param_file_path, 'r') as file:
        data = json.load(file)
        return data