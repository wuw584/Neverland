import os
import pandas as pd
import numpy as np
import cv2

def txt2line(txt_dir,patch_dir):
    color = [(0,255,0),(255,213,0)]

    txt_names = os.listdir(txt_dir)
    patch_names = [n.replace("txt","png") for n in txt_names]

    for i in range(len(txt_names)):
        t_name = txt_names[i]
        p_name = patch_names[i]
        txt_path = os.path.join(txt_dir,t_name)
        patch_path = os.path.join(patch_dir,p_name)
        patch_img = cv2.imread(patch_path)
        img_h,img_w = patch_img.shape[:2]
        with open(txt_path, "r") as file:
            box_data = file.readlines()
        for j in range(len(box_data)):
            single_box = box_data[j][:-1]   # 最后一个是换行符，去掉
            single_box_str = single_box.split(" ")
            single_box_float = [float(s) for s in single_box_str]

            label = single_box_float[0]
            xywh = single_box_float[1:]
            xyxy = xywh2xyxy(img_w,img_h,xywh)

            if label==0:
                p1 = (xyxy[2],xyxy[1])
                p2 = (xyxy[0],xyxy[3])
                c = color[0]
            elif label==1:
                p1 = (xyxy[0],xyxy[1])
                p2 = (xyxy[2],xyxy[3])
                c = color[1]

            cv2.line(patch_img,p1,p2,c,2)

        cv2.imwrite(patch_path,patch_img)

#tool
def xywh2xyxy(img_w,img_h,xywh):
    x,y,w,h = xywh
    x1 = round((x-0.5*w)*img_w-0.5)
    x2 = round((x+0.5*w)*img_w-0.5)

    y1 = round((y-0.5*h)*img_h-0.5)
    y2 = round((y+0.5*h)*img_h-0.5)

    return [x1,y1,x2,y2]

def box2line(txt_path):
    #输出：
    #p1:start point
    #p2:end point
    with open(txt_path, "r") as file:
        box_data = file.readlines()
    lines = []
    for j in range(len(box_data)):
        single_box = box_data[j][:-1]   # 最后一个是换行符，去掉
        single_box_str = single_box.split(" ")
        single_box_float = [float(s) for s in single_box_str]

        label = single_box_float[0]
        xywh = single_box_float[1:]
        xyxy = xywh2xyxy(96,96,xywh)

        if label==0:
            p1 = [xyxy[2],xyxy[1]]
            p2 = [xyxy[0],xyxy[3]]
            c = 0
        elif label==1:
            p1 = [xyxy[0],xyxy[1]]
            p2 = [xyxy[2],xyxy[3]]
            c = 1
        lines.append([p1,p2,c])
    return lines


#对于从yolo的输出文件夹中的一系列txt文档中提取出完整的长线段，输出到一个txt文件中
def txt2json(img_name,yolo_dir,txt_save_path,img_w,img_h ,padding=16 , patch_num = 75 , patch_size=64):
    line_list = []
    patch_num = int(img_h / patch_size) 
    count = 0
    for i in range(int(img_w / patch_size)):
        for j in range(int(img_h / patch_size)):
            patch_id = i*patch_num+j
            # print(patch_id)
            patch_name = img_name + '_' + str(patch_id).zfill(5) + '.txt'
            patch_path = os.path.join(yolo_dir, patch_name)
            if os.path.exists(patch_path):
                lines= box2line(patch_path)
                for line in lines:
                    p1,p2,c = line
                    p1 = list(np.add(p1,  [j * patch_size - padding,i * patch_size - padding]))
                    p2 = list(np.add(p2,  [j * patch_size - padding,i * patch_size - padding]))
                    line_list.append([p1,p2,c])
    print(len(line_list))
    #TODO 对line_list中的线段进行关联

    

    # 将线段写入对应的txt文件
    print(len(line_list))
    for line in line_list:
        txt_content = '-'.join([str(a) for a in line]) + '\n'
        with open(txt_save_path, 'a') as txt_file:
            txt_file.write(txt_content)
            
#将名称为img_name的小块图片拼接成大图
def patch2img_pad(patch_dir,img_name,img_save_path,img_w,img_h ,padding=16 , patch_num = 75 , patch_size=64):
    patch_size = 64
    img = np.zeros((img_w,img_h,3),dtype=np.uint8)
    patch_num = int(img_h / patch_size) 
    for i in range(int(img_w / patch_size)):
        for j in range(int(img_h / patch_size)):
            patch_id = i*patch_num+j
            patch_name = img_name + '_' + str(patch_id).zfill(5) + '.png'
            patch_path = os.path.join(patch_dir, patch_name)
            patch_img = cv2.imread(patch_path)
            img[i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size] = patch_img[padding:padding+patch_size,padding:padding+patch_size]
    cv2.imwrite(img_save_path,img)

#将txt文件的线段画到图片中，之后可以改为将json文件的画到大图中
def draw_line(img_dir):
    color = [(0,255,0),(255,213,0)]
    file_list = os.listdir(img_dir)
    img_list = [f for f in file_list if f.endswith("png")]
    img_list.sort()
    txt_list = [f for f in file_list if f.endswith("txt")]
    txt_list.sort()
    for i in range(len(txt_list)):
        img_name = img_list[i]
        txt_name = txt_list[i]
        img_path = os.path.join(img_dir,img_name)
        txt_path = os.path.join(img_dir,txt_name)
        img = cv2.imread(img_path)
        with open(txt_path, "r") as file:
            lines = file.readlines()
        for j in range(len(lines)):
            single_line = lines[j][:-1]   # 最后一个是换行符，去掉
            single_line_str = single_line.split("-")
            p1 = eval(single_line_str[0])
            p2 =eval(single_line_str[1])
            c = color[int(single_line_str[2])]
            cv2.line(img,p1,p2,c,2)
        cv2.imwrite(img_path,img)


#TODO 后续对车流线的分析可以进一步保存在json文件中