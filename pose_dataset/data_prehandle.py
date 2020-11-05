import cv2
import numpy as np
import os
import json

def cv_imread(filedir):
    cv_img = cv2.imdecode(np.fromfile(filedir,dtype=np.uint8),-1)
    return cv_img

# 数据预处理
# 动作标签集
action_label = ["normal","left","right","behind","down"]

# 关键点标签集
position_label = ["earl","eyel","nose","eyer","earr","mouth","jaw","neck","bearl","bearr","hug1","hug2","hug3"]

# 动作统计
action_count_dict = {"normal": 0,"left": 0,"right": 0,"behind": 0,"down": 0}

# 数据清洗
def pose_keypoints_show(jsonroot,imgroot):

    img_folder = ['front','left','right']


    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255],
              ]

    pairs = [[0,1],[1,2],[2,3],[3,4],[2,5],[5,6],[6,7],[7,10],[10,11],[11,12],[7,8],[7,9]]
    colors_skeleton = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0],
                       [0, 255, 0],
                       [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255],
                       ]

    for file in os.listdir(f"{jsonroot}"):
        with open(f"{jsonroot}/{file}",'r') as f:
            data = json.load(f)
            pose = np.zeros((13,2),dtype=np.int32)
            key_offset = {}
            data = data['shapes']

            for item in data:
                if item['label'] not in key_offset:
                    key_offset[item['label']] = item['points'][0]
                if item['shape_type']== "rectangle":
                    pose_type = item['label']
                    rect = item['points']
            try:
                for i,key_name in enumerate(position_label):
                    pose[i] = np.array(key_offset[key_name])
            except Exception:
                print(Exception,file)
                continue
            # print(pose)
            img_index = file.split('.')[0]
            if  int(img_index) <= 1194:
                folder = 'front'
            elif  int(img_index) <= 2337 and int(img_index)>=1195:
                folder = 'left'
            else:
                folder = 'right'


            img = cv_imread(f"{imgroot}/{folder}/{img_index}.jpg")

            for idx in range(len(colors)):
                cv2.circle(img, (pose[idx, 0], pose[idx, 1]), 3, colors[idx], thickness=1, lineType=8, shift=0)
            for idx in range(len(colors_skeleton)):
                img = cv2.line(img, (pose[pairs[idx][0], 0], pose[pairs[idx][0], 1]),
                                 (pose[pairs[idx][1], 0], pose[pairs[idx][1], 1]), colors_skeleton[idx], thickness=2,)
            cv2.rectangle(img,(int(rect[0][0]),int(rect[0][1])),(int(rect[1][0]),int(rect[1][1])),(0,0,255),thickness=2)


            cv2.imshow(f"{img_index}.jpg:"+pose_type,img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

if __name__ == '__main__':


    pose_keypoints_show(r'z:\第六组\坐姿矫正提示系统\数据\pose_image\label',r'z:\第六组\坐姿矫正提示系统\数据\pose_image')

