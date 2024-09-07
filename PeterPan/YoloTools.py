import os
import pandas as pd
import numpy as np
import cv2
import math


from scipy.optimize import linear_sum_assignment


def line2box(line):
    p1,p2,c = line
    if c==0: #从图像上看 斜率大于零 pos
        xyxy = [p2[0],p1[1],p1[0],p2[1]]
        # p1 = [xyxy[2],xyxy[1]] #右上角x y
        # p2 = [xyxy[0],xyxy[3]] #左下角x y
    elif c==1: #斜率小于零 neg
        xyxy = [p1[0],p1[1],p2[0],p2[1]]
        # p1 = [xyxy[0],xyxy[1]] #左上角x y
        # p2 = [xyxy[2],xyxy[3]] #右下角x y 
    return xyxy


def iou(bb_test, bb_gt):
    """
    在两个box间计算IOU
    :param bb_test: box1 = [x1y1x2y2] 即 [左上角的x坐标，左上角的y坐标，右下角的x坐标，右下角的y坐标]
    :param bb_gt: box2 = [x1y1x2y2]
    :return: 交并比IOU
    """
    xx1 = np.maximum(bb_test[0], bb_gt[0]) #获取交集面积四边形的 左上角的x坐标
    yy1 = np.maximum(bb_test[1], bb_gt[1]) #获取交集面积四边形的 左上角的y坐标
    xx2 = np.minimum(bb_test[2], bb_gt[2]) #获取交集面积四边形的 右下角的x坐标
    yy2 = np.minimum(bb_test[3], bb_gt[3]) #获取交集面积四边形的 右下角的y坐标
    w = np.maximum(0., xx2 - xx1) #交集面积四边形的 右下角的x坐标 - 左上角的x坐标 = 交集面积四边形的宽
    h = np.maximum(0., yy2 - yy1) #交集面积四边形的 右下角的y坐标 - 左上角的y坐标 = 交集面积四边形的高
    wh = w * h #交集面积四边形的宽 * 交集面积四边形的高 = 交集面积
    """
    两者的交集面积，作为分子。
    两者的并集面积作为分母。
    一方box框的面积：(bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
    另外一方box框的面积：(bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) 
    """
    o = wh / ( (bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
               + (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1])
               - wh)
    return o


def line_iou(ls_test, ls_gt):
    bb_test = line2box(ls_test)
    bb_gt = line2box(ls_gt)
    return iou(bb_test , bb_gt )


def metric(ls_test, ls_gt): #ls = line segment
    """
    在两个线段之间计算匹配值 
    :param ls_test: line 1 = [p1 , p2 , c ] 即 [向量起点[x,y] , 向量终点 , 方向] (新一帧 , 检测框)
    :param ls_gt: line 2 = [p1 , p2 , c ] 即 [向量起点 , 向量终点 , 方向] (旧一帧 , 原目标)
    :return: 交并比IOU = k1 * cos(向量夹角) +  k2 * dis( 目标终点 , 检测起点 ) + k3 * (c1 == c2) + k4 * iou
    期望匹配值越大越好，即
        夹角越小越好 cos越大越好 k1 > 0 
        距离越小越好 k2 < 0  
        方向一致 k3 > 0
        抵消相交导致的距离过大 k4 > 0
    """
    k1 , k2 , k3 , k4 = 1 , -0.2 , 1 , 10 
    va = np.array( [ls_test[1][0] - ls_test[0][0] , ls_test[1][1] - ls_test[0][1] ])
    vb = np.array( [ls_gt[1][0] - ls_gt[0][0] , ls_gt[1][1] - ls_gt[0][1] ])
    cos = va.dot(vb) / ( np.sqrt(va.dot(va)) * np.sqrt(vb.dot(vb)))

    gt_end = ls_gt[1]
    test_start = ls_test[0]
    dis1 = math.sqrt(math.pow((gt_end[0]- test_start[0]),2)+math.pow((gt_end[1]-test_start[1]),2))

    gt_end = ls_gt[0]
    test_start = ls_test[1]
    dis2 = math.sqrt(math.pow((gt_end[0]- test_start[0]),2)+math.pow((gt_end[1]-test_start[1]),2))

    dis = min(dis1,dis2) #考虑到斜率为正值 会出现向量方向和拼接方向相反的情况，dis 改为算线段最短端点距离

    cost =  k1 * cos + k2 * dis + k3 * (ls_test[2] == ls_gt[2]) 
    # o = line_iou(ls_test, ls_gt) #加入交并比 改变不大 反而计算更慢
    # cost =  k1 * cos + k2 * dis + k3 * (ls_test[2] == ls_gt[2]) + k4 * o 
    # if dis < 10 :
    #     print( "o" ,o , cost , dis , dis1 , dis2  )
    return cost


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    将检测框bbox与卡尔曼滤波器的跟踪框进行关联匹配
    :param detections:检测框(新一帧检测到的) 
    :param trackers:跟踪框，即跟踪目标(原有的目标)
    :param iou_threshold:IOU阈值
    :return:跟踪成功目标的矩阵：matchs， 返回的是下标 不是数组本身
            新增目标的矩阵：unmatched_detections
            跟踪失败即离开画面的目标矩阵：unmatched_trackers
    """
    # 跟踪目标数量为0，直接构造结果
    if (len(trackers) == 0) or (len(detections) == 0):
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

    # iou 不支持数组计算。逐个计算两两间的交并比，调用 linear_assignment 进行匹配
    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
    # 遍历目标检测的bbox集合，每个检测框的标识为d
    for d, det in enumerate(detections):
        # 遍历跟踪框（卡尔曼滤波器预测）bbox集合，每个跟踪框标识为t
        for t, trk in enumerate(trackers):
            iou_matrix[d, t] = metric(det, trk)
    # 通过匈牙利算法将跟踪框和检测框以[[d,t]...]的二维矩阵的形式存储在match_indices中
    result = linear_sum_assignment(-iou_matrix)
    matched_indices = np.array(list(zip(*result)))
 
    # 记录未匹配的检测框及跟踪框
    # 未匹配的检测框放入unmatched_detections中，表示有新的目标进入画面，要新增跟踪器跟踪目标
    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
    # 未匹配的跟踪框放入unmatched_trackers中，表示目标离开之前的画面，应删除对应的跟踪器
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)
    # 将匹配成功的跟踪框放入matches中
    matches = []
    for m in matched_indices:
        # 过滤掉IOU低的匹配，将其放入到unmatched_detections和unmatched_trackers
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        # 满足条件的以[[d,t]...]的形式放入matches中
        else:
            matches.append(m.reshape(1, 2))
    # 初始化matches,以np.array的形式返回
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)
 
    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


#tool
def xywh2xyxy(img_w,img_h,xywh):
    """框中心x,y 框宽 框高"""
    x,y,w,h = xywh
    x1 = round((x-0.5*w)*img_w-0.5)
    x2 = round((x+0.5*w)*img_w-0.5)

    y1 = round((y-0.5*h)*img_h-0.5)
    y2 = round((y+0.5*h)*img_h-0.5)

    return [x1,y1,x2,y2]

#把txt文档转换成 线段
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

        if label==0: #从图像上看 斜率大于零 pos
            p1 = [xyxy[2],xyxy[1]] #右上角x y
            p2 = [xyxy[0],xyxy[3]] #左下角x y
            c = 0
        elif label==1: #斜率小于零 neg
            p1 = [xyxy[0],xyxy[1]] #左上角x y
            p2 = [xyxy[2],xyxy[3]] #右下角x y 
            c = 1
        lines.append([p1,p2,c])
   
    return lines


#对于从yolo的输出文件夹中的一系列txt文档中提取出完整的长线段，输出到一个txt文件中
def txt2json(img_name,yolo_dir,txt_save_path,img_h,img_w ,padding=16 , patch_num = 75 , patch_size=64 , iou_threshold = 0.3 , min_seg_num = 2):
    patch_num = int(img_w / patch_size) 
    count = 0
    trackers = []
    for i in range(int(img_h / patch_size)): #从上到下
        for j in range(int(img_w / patch_size)): #从左到右的方式关联
            detections = []
            patch_id = i * patch_num + j
            patch_name = img_name + '_' + str(patch_id).zfill(5) + '.txt'
            patch_path = os.path.join(yolo_dir, patch_name)
            if os.path.exists(patch_path):
                lines= box2line(patch_path)
                for line in lines:
                    p1,p2,c = line
                    p1 = list(np.add(p1,  [j * patch_size - padding,i * patch_size - padding]))
                    p2 = list(np.add(p2,  [j * patch_size - padding,i * patch_size - padding]))
                    detections.append([p1,p2,c])
                    count += 1

            # TODO 先横向拼接，再纵向
            if (len(trackers) != 0) and (len(detections) != 0):
                #对新一帧图片中包含的线段进行关联
                # print("line_end_node",  [line[-1] for line in  trackers])
                matches, unmatched_detections,unmatched_trackers = associate_detections_to_trackers( detections ,  [line[-1] for line in  trackers] , iou_threshold)
                #update trackers
                for d,t in matches:
                    trackers[t].append(detections[d])
            else:  
                # 跟踪目标数量为0，直接构造结果
                unmatched_detections =  np.arange(len(detections))
            for d in unmatched_detections:
                trackers.append([detections[d]])
            # print(trackers)

    #对关联结果简单评估
    # trackers1 = [line for line in trackers if len(line) >= 1 ] #关联线段数> 3 才存入txt
    # trackers2 = [line for line in trackers if len(line) >= 2 ] #关联线段数> 3 才存入txt
    # trackers3 = [line for line in trackers if len(line) >= 3 ] #关联线段数> 3 才存入txt
    # print(img_name, "\tfrom   total  #" , count , "  \tto  1#" , len(trackers1), "  \tto  2#" , len(trackers2), "  \tto  3#" , len(trackers3))

    # #筛选可用长线段
    trackers = [line for line in trackers if len(line) >= min_seg_num ] #关联线段数> 3 才存入txt

    # 将长线段写入对应的txt文件
    for line in trackers:            
        # txt_content = ','.join([str(a) for a in line]) + '\n'
        with open(txt_save_path, 'a') as txt_file:
            txt_file.write(str(line)+ '\n')
            
#将名称为img_name的小块图片拼接成大图
def patch2img_pad(patch_dir,img_name,img_save_path,img_h,img_w ,padding=16 , patch_num = 75 , patch_size=64):
    patch_size = 64
    img = np.zeros((img_h,img_w,3),dtype=np.uint8)
    patch_num = int(img_w / patch_size) 
    for i in range(int(img_h / patch_size)): #上到下
        for j in range(int(img_w / patch_size)): #左到右
            patch_id = i*patch_num+j
            patch_name = img_name + '_' + str(patch_id).zfill(5) + '.png'
            patch_path = os.path.join(patch_dir, patch_name)
            patch_img = cv2.imread(patch_path)
            img[i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size] = patch_img[padding:padding+patch_size,padding:padding+patch_size]
            #画出分割成小图的边界线
            cv2.rectangle(img,  # 图片
					 (j * patch_size, i * patch_size),  # (xmin, ymin)左上角坐标
					 ((j + 1) * patch_size , (i + 1) * patch_size),  # (xmax, ymax)右下角坐标
					 (0, 255, 0), 1)  # 颜色，线条宽度
            # cv2.line(patch_img,p1,p2,(255,213,0),1)

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
            # single_line_str = single_line.split("-")
            trace_list = eval(single_line)
            points = []
            for line in trace_list:
                points.append(line[0])
                points.append(line[1])
            points = np.array(points)
            pts_fit3 = np.polyfit(points[:, 0], points[:, 1], 3)  # 拟合为三次曲线
            p1 = np.poly1d(pts_fit3) #使用次数合成多项式
            y_pre = p1(points[:, 0]) # 得到三次曲线对应的点集
            y_pre = points.reshape(-1,1,2)
            color = tuple([int(x) for x in np.random.choice(range(256), size=3).astype(np.uint8)])#随机颜色
            cv2.polylines(img, np.int32([y_pre]), False, color, 3)  #三次曲线上的散点构成的折线图，近似为曲线

        cv2.imwrite(img_path,img)


#TODO 后续对车流线的分析可以进一步保存在json文件中