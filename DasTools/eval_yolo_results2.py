import os
import numpy as np
import cv2


def img2patch_pad(img_path,save_dir,patch_size=64,padding=16):
    # 将整张4800*4800的图切割为64*64,以行列序列化尾缀命名
    img_name = img_path.split("/")[-1].replace(".png","")
    # print(img_path)
    img = cv2.imread(img_path)
    img_w,img_h = img.shape[:2] 
    # print(img_w,img_h)    # 4800,4800   
    patch_num = int(img_h / patch_size)  # 75

    for i in range(int(img_w / patch_size)):
        for j in range(int(img_h / patch_size)):
            patch_img = img[i*patch_size:(i+1)*patch_size,j*patch_size:(j+1)*patch_size]
            patch_id = i*patch_num+j
            patch_name = img_name + '_' + str(patch_id).zfill(5) + '.png'
            patch_path = os.path.join(save_dir,patch_name)
            patch_img_pad = cv2.copyMakeBorder(patch_img, padding, padding, padding, padding, cv2.BORDER_CONSTANT, 128)
            cv2.imwrite(patch_path,patch_img_pad)
    # print("done",patch_path)
    return patch_num,img_w,img_h 


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

def xywh2xyxy(img_w,img_h,xywh):
    x,y,w,h = xywh
    x1 = round((x-0.5*w)*img_w-0.5)
    x2 = round((x+0.5*w)*img_w-0.5)

    y1 = round((y-0.5*h)*img_h-0.5)
    y2 = round((y+0.5*h)*img_h-0.5)

    return [x1,y1,x2,y2]

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



# --------------可视化验证流程, 4个步骤依次运行-------------------

# img_path = r'../Highway_2024_0510/output/0628_data2/deci_7_7/pic/0.png' 
# patch_dir =    r'../Highway_2024_0510/output/0628_data2/deci_7_7/patches'
# yolo_predict = r'../Highway_2024_0510/output/0628_data2/deci_7_7/yolo/labels'   # 每次预测完会存入新的文件夹，需修改“exp”对应的数字
# img_save_dir = r"../Highway_2024_0510/output/0628_data2/deci_7_7/predict_vis_result"

# # 1.img to patch, 待验证图像划分patches保存
# if not os.path.exists(patch_dir):
#     os.makedirs(patch_dir)
# img2patch_pad(img_path, patch_dir, patch_size=64, padding=16)

# # 2. yolo predict, 用训练好的yolo模型预测上一步骤中的patches
# # 命令：python detect.py --weights ./runs/train/exp7/weights/best.pt --source ../datasets/traffic/images/val --imgsz 192 --iou-thres 0.5 --conf-thres 0.5 --save-txt
# # 命令：python detect.py --weights ./runs/train/exp7/weights/best.pt --source ../Highway_2024_0510/output/0628_data2/deci_7_7/patches --project '../Highway_2024_0510/output/0628_data2/' --name 'deci_7_7/yolo/' --imgsz 192 --iou-thres 0.5 --conf-thres 0.5 --save-txt

# # 3. draw predicted lines to patch, 将预测结果绘制到每一patch上
# txt2line(yolo_predict,patch_dir)

# # 4. patch to img, 将绘制了预测结果的patch合并为大图像并保存
# img_name = img_path.split("/")[-1].replace(".png","")
# img_save_path = os.path.join(img_save_dir,img_name+'.png')
# patch2img_pad(patch_dir,img_name, img_save_path, padding=16)