import os
import shutil
import pandas as pd
import json
import numpy as np
import cv2
import math
import random

def bbox2yolo(img_w, img_h, bbox):
    # bbox:(x1,y1,x2,y2)
    x_center = ((bbox[0] + bbox[2]) / 2 + 0.5) / img_w
    y_center = ((bbox[1] + bbox[3]) / 2 + 0.5) / img_h
    w = abs(bbox[2] - bbox[0]) / img_w
    h = abs(bbox[3] - bbox[1]) / img_h
    yolo_bbox = [x_center, y_center, w, h]
    return yolo_bbox

def line_devide(line=None,patch_size=None,patch_num=None,img_name="EW_fiber_0"):
    # 输入大图中的单根线，将其划分到各个patch中
    # 算法逻辑：找到所有跟轴的交点，再计算所属图像
    # 输入：
    # -- line: [x1,y1,x2,y2],长线两端点坐标
    # -- patch_size: N = 64
    # -- patch_num: 4800 / patch_size = 75
    # -- img_name: 大图名

    # 输出：
    # -- lines: 划分好后的小线段列表
    # -- img_names: 小线段对于归属的patch名

    x1,y1,x2,y2 = line
    if x1 < x2:
        x_l = x1
        y_l = y1
        x_r = x2
        y_r = y2
    else:
        x_l = x2
        y_l = y2
        x_r = x1
        y_r = y1
    
    # 斜率
    k = (y_r-y_l)/(x_r-x_l)

    # 找到长线与所有x方向网格线的交点
    mid_nodes_x = (np.arange(math.floor(x_l/patch_size),math.floor(x_r/patch_size))+1)*patch_size
    mid_nodes_x = np.concatenate((mid_nodes_x - 1, mid_nodes_x))
    mid_nodes_x = np.sort(mid_nodes_x)

    # 找到长线与所有y方向网格线的交点
    if y_l < y_r:
        mid_nodes_y = (np.arange(math.floor(y_l/patch_size),math.floor(y_r/patch_size))+1)*patch_size
        mid_nodes_y = np.concatenate((mid_nodes_y-1,mid_nodes_y))
        mid_nodes_y = np.sort(mid_nodes_y)

    else:

        mid_nodes_y = (np.arange(math.floor(y_r / patch_size), math.floor(y_l / patch_size))+1) * patch_size
        mid_nodes_y = np.concatenate((mid_nodes_y-1,mid_nodes_y))
        mid_nodes_y = np.sort(mid_nodes_y)
        mid_nodes_y = np.flip(mid_nodes_y)

    # 将所有与y网格线的交点求对应的x值
    mid_nodes_y2x =[((y-y_l)/k+x_l) for y in mid_nodes_y]

    # 获得所有网格线（包括x和y方向）的交点的x坐标，排序，再获得对应的y坐标，
    x_nodes = np.concatenate(([x_l],mid_nodes_x,[x_r],mid_nodes_y2x))
    x_nodes = np.sort(x_nodes)
    y_nodes = np.array([(x-x_l)*k+y_l for x in x_nodes])

    lines = []
    img_names = []
    line_n = int(len(x_nodes) / 2)

    # 由网格线交点坐标，将小线段划分到对应的patch中
    for i in range(line_n):
        x1 = x_nodes[i*2]+0.01
        x2 = x_nodes[i*2+1]-0.01
        y1 = y_nodes[i*2]+0.01
        y2 = y_nodes[i*2+1]-0.01

        img_x = math.floor(x1/patch_size)
        img_y = math.floor(y1/patch_size)

        x1 = x1 - (img_x * patch_size)
        x2 = x2 - (img_x * patch_size)

        y1 = y1 - (img_y * patch_size)
        y2 = y2 - (img_y * patch_size)
        
        # 计算patch编号
        img_num = img_y*patch_num+img_x

        i_name = img_name + '_' + str(img_num).zfill(5)
        lines.append([x1,y1,x2,y2])
        img_names.append(i_name)

    return lines,img_names

def img2patch_pad(img_path,save_dir,patch_size=64,padding=16):
    # 将整张4800*4800的图切割为64*64,以行列序列化尾缀命名
    img_name = img_path.split("/")[-1].replace(".png","")
    print(img_name)
    img = cv2.imread(img_path)
    img_w,img_h = img.shape[:2]     # 4800,4800
    patch_num = int(img_h / patch_size)  # 4800/64 = 75

    for i in range(int(img_w / patch_size)):
        for j in range(int(img_h / patch_size)):
            patch_img = img[i*patch_size:(i+1)*patch_size,j*patch_size:(j+1)*patch_size]
            patch_id = i*patch_num+j
            patch_name = "pic5_1_"+img_name + '_' + str(patch_id).zfill(5) + '.png'
            patch_path = os.path.join(save_dir,patch_name)

            patch_img_pad = cv2.copyMakeBorder(patch_img, padding, padding, padding, padding, cv2.BORDER_CONSTANT)
            cv2.imwrite(patch_path,patch_img_pad)

def xlabel2yolo_pad(img_name,xlabel_json_path,yolo_save_dir,patch_size,padding):
    with open(xlabel_json_path, 'r') as f:
        json_data = json.load(f)

    img_w = patch_size + (2*padding)
    img_h = patch_size + (2*padding)
    shapes = json_data["shapes"]

    L = len(shapes)
    print(f"num of lines:{L}")
    for i in range(L):
        single_object = shapes[i]
        x1 = single_object["points"][0][0]
        y1 = single_object["points"][0][1]
        x2 = single_object["points"][1][0]
        y2 = single_object["points"][1][1]

        long_line = [x1,y1,x2,y2]

        # 将line划分到各个patch中
        patch_lines,patch_names = line_devide(long_line, patch_size=64, patch_num=25, img_name=img_name)
        for j in range(len(patch_lines)):
            yolo_txt_name = "pic5_1_"+patch_names[j] + '.txt'
            yolo_txt_path = os.path.join(yolo_save_dir,yolo_txt_name)

            p_line = patch_lines[j]

            # 根据线段斜率划分车流方向
            if (p_line[3]-p_line[1])/(p_line[2]-p_line[0])<0:
                label = 0       # line_l
            else:
                label = 1       # line_r

            p_line = patch_lines[j]
            p_line = [v+padding for v in p_line]            # 加上padding
            bbox_yolo = bbox2yolo(img_w, img_h, p_line)     # 转yolo格式，[x1,y1,x2,y2]->[x,y,w,h]

            # 写入对应的yolo标注文件中
            txt_content = str(label) + ' ' + ' '.join([str(a) for a in bbox_yolo]) + '\n'
            with open(yolo_txt_path, 'a') as txt_file:
                txt_file.write(txt_content)

def devide_dataset(label_src_dir,
                   img_src_dir,
                   train_label_save_dir,
                   train_img_save_dir,
                   val_label_save_dir,
                   val_img_save_dir,
                   tarin_ratio=0.9,
                   with_neg=False):

    # 将patch img和label划分yolo训练集和测试集
    # 输入:
    # --label_src_dir: label原始文件夹
    # --img_src_dir: img原始文件夹
    # --train_label_save_dir: 划分后训练集label问价夹
    # --train_img_save_dir: 划分后训练集img问价夹
    # --val_label_save_dir: 划分后验证集label文件夹
    # --val_img_save_dir: 划分后验证集img文件夹
    # --tarin_ratio: 训练集占比例
    # --with_neg: 是否将无标签的数据纳入作为负样本

    img_names = os.listdir(img_src_dir)
    names = [n.split(".")[0] for n in img_names]
    random.seed(7)
    random.shuffle(names)

    tarin_num = round(len(names)*tarin_ratio)
    train_names = names[:tarin_num]
    val_names = names[tarin_num:]

    # 训练集划分
    for i in range(len(train_names)):
        name = train_names[i]

        img_path = os.path.join(img_src_dir,name+'.png')
        txt_path = os.path.join(label_src_dir,name+'.txt')

        img_save = os.path.join(train_img_save_dir,name+'.png')
        txt_save = os.path.join(train_label_save_dir,name+'.txt')

        # copy
        if(os.path.isfile(txt_path)):
            shutil.copy(img_path, img_save)
            shutil.copy(txt_path,txt_save)
        elif with_neg:
            shutil.copy(img_path, img_save)
            with open(txt_save,"w") as file:
                pass
        else:
            pass

    # 验证集划分
    for i in range(len(val_names)):
        name = val_names[i]

        img_path = os.path.join(img_src_dir, name + '.png')
        txt_path = os.path.join(label_src_dir, name + '.txt')

        img_save = os.path.join(val_img_save_dir, name + '.png')
        txt_save = os.path.join(val_label_save_dir, name + '.txt')

        # 验证集无需负样本
        if (os.path.isfile(txt_path)):
            shutil.copy(txt_path, txt_save)
            shutil.copy(img_path, img_save)
        else:
            pass


def get_patches(data_dir,patch_save_dir,yolo_save_dir):
    # 将标注好的数据切块，计算对应的yolo标注，并保存
    # 输入
    # --data_dir: 已标注图像及json的文件夹
    # --patch_save_dir: 划分好后的img patch保存路径
    # --yolo_save_dir: 划分好后的yolo 标签 保存路径
    # 遍历处理问价夹下所有图片及标注
    file_list = os.listdir(data_dir)
    img_list = [f for f in file_list if f.endswith("png")]
    img_list.sort()
    json_list = [f for f in file_list if f.endswith("json")]
    json_list.sort()
    print(img_list)
    for i in range(len(img_list)):
        img_name = img_list[i]
        i_name = img_name.replace(".png","")
        json_name = json_list[i]
        img_path = os.path.join(data_dir,img_name)
        json_path = os.path.join(data_dir,json_name)
        patch_size = 64
        padding = 16
        # 1. 将大图分成小块，并填充边缘。小块大小为（64，64），边缘填充16像素黑边
        img2patch_pad(img_path, patch_save_dir, patch_size,padding)

        # 2. 将xlabel标注的大图数据划分到每一小图中，并转换为yolo标注格式
        xlabel2yolo_pad(i_name, json_path, yolo_save_dir,patch_size,padding)


# -----------------------步骤一：获取patch和对应的yolo label----------------------------
# data_dir =r"xxxxx"
# patch_save_dir = r"xxxxx"
# yolo_save_dir = r"xxxxx"
# get_patches(data_dir,patch_save_dir,yolo_save_dir)


# ---------------步骤二：划分训练集和测试集----------------------------------
# label_src_dir = r"xxxxx"
# img_src_dir = r"xxxxx"
# train_label_save_dir = r"xxxxx"
# val_label_save_dir = r"xxxxx"
# train_img_save_dir = r"xxxxx"
# val_img_save_dir = r"xxxxx"
# ratio = 0.9
# with_neg = False
# devide_dataset(label_src_dir,
#                 img_src_dir,
#                 train_label_save_dir,
#                 train_img_save_dir,
#                 val_label_save_dir,
#                 val_img_save_dir,
#                 ratio,
#                 with_neg)

