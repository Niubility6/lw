# from scipy.io import loadmat
# import numpy as np
#
#
# def save_joints():
#     mat = loadmat(r'D:\迅雷下载\MobilePose-pytorch-master\pose_dataset\mpii\mpii_human_pose_v1_u12_1.mat')
#     fout = open(r'1.txt', 'w')
#     for i, (anno, train_flag) in enumerate(
#             zip(mat['RELEASE']['annolist'][0, 0][0],
#                 mat['RELEASE']['img_train'][0, 0][0],
#                 )):
#         img_fn = anno['image']['name'][0, 0][0]
#         train_flag = int(train_flag)
#
#         head_rect = []
#         if 'x1' in str(anno['annorect'].dtype):
#             head_rect = zip(
#                 [x1[0, 0] for x1 in anno['annorect']['x1'][0]],
#                 [y1[0, 0] for y1 in anno['annorect']['y1'][0]],
#                 [x2[0, 0] for x2 in anno['annorect']['x2'][0]],
#                 [y2[0, 0] for y2 in anno['annorect']['y2'][0]])
#
#         if 'annopoints' in str(anno['annorect'].dtype):
#             # only one person
#             annopoints = anno['annorect']['annopoints'][0]
#             head_x1s = anno['annorect']['x1'][0]
#             head_y1s = anno['annorect']['y1'][0]
#             head_x2s = anno['annorect']['x2'][0]
#             head_y2s = anno['annorect']['y2'][0]
#
#             for annopoint, head_x1, head_y1, head_x2, head_y2 in zip(
#                     annopoints, head_x1s, head_y1s, head_x2s, head_y2s):
#                 if annopoint != []:
#                     head_rect = [float(head_x1[0, 0]),
#                                  float(head_y1[0, 0]),
#                                  float(head_x2[0, 0]),
#                                  float(head_y2[0, 0])]
#                     # build feed_dict
#                     feed_dict = {}
#                     feed_dict['width'] = int(abs(float(head_x2[0, 0]) - float(head_x1[0, 0])))
#                     feed_dict['height'] = int(abs(float(head_y2[0, 0]) - float(head_y1[0, 0])))
#
#                     # joint coordinates
#                     annopoint = annopoint['point'][0, 0]
#                     j_id = [str(j_i[0, 0]) for j_i in annopoint['id'][0]]
#                     x = [x[0, 0] for x in annopoint['x'][0]]
#                     y = [y[0, 0] for y in annopoint['y'][0]]
#                     joint_pos = {}
#                     for _j_id, (_x, _y) in zip(j_id, zip(x, y)):
#                         joint_pos[str(_j_id)] = [float(_x), float(_y)]
#
#                     # visiblity list
#                     if 'is_visible' in str(annopoint.dtype):
#                         vis = [v[0] if v else [0]
#                                for v in annopoint['is_visible'][0]]
#                         vis = dict([(k, int(v[0])) if len(v) > 0 else v
#                                     for k, v in zip(j_id, vis)])
#                     else:
#                         vis = None
#                     feed_dict['x'] = x
#                     feed_dict['y'] = y
#                     feed_dict['vis'] = vis
#                     feed_dict['filename'] = img_fn
#
#                     if len(joint_pos) == 16:
#                         data = {
#                             'filename': img_fn,
#                             'train': train_flag,
#                             'head_rect': head_rect,
#                             'is_visible': vis,
#                             'joint_pos': joint_pos
#                         }
#
#             print(data)
#
#             label = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15']
#             sss = ' '
#             for key in label:
#                 sss = sss + str(int(data['joint_pos'][key][0])) + ' ' + str(int(data['joint_pos'][key][1])) + ' ' + str(
#                     int(data['is_visible'][key])) + ' '
#             sss = sss.strip()
#             fout.write(data['filename'] + ' ' + sss + '\n')
#     fout.close()
#
#
# if __name__ == '__main__':
#     save_joints()
import numpy as np
import cv2
import os

if __name__ == '__main__':
    dst = r"./ooo"
    img_name = r"093311333.jpg"
    label = "710,409,581,468,687,354,779,334,734,348,750,443,733,344,721,150,716.3796,140.0421,667.6204,34.9579,598,102,555,71,657,139,784,161,800,301,681,320"
    label = label.split(',')

    label = [float(temp) for temp in label]
    label = np.array(label).reshape(16, 2)
    print(label)
    img = cv2.imread(os.path.join("pose_dataset/mpii/images",img_name))

    for i, key in enumerate(label):

        cv2.circle(img, (int(key[0]), int(key[1])), 1, (0, 0, 255), 2)
        cv2.putText(img, str(i), (int(key[0]), int(key[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)

    cv2.imwrite(f"{dst}/{0}.jpg",img)
    cv2.imshow('src', img)
    cv2.waitKey(0)