import os
import time
import argparse
import numpy as np
import paramiko
import subprocess
import tracker
from detector import Detector
import cv2

# 定义全局变量用于拿去下载文件的路径
glob_file_path=''

"""
实现了 出/入 分别计数。
显示检测类别。
默认是 南/北 方向检测，若要检测不同位置和方向，可在 main.py 文件第28行和36行，修改2个polygon的点。
默认检测类别：行人、自行车、小汽车、摩托车、公交车、卡车。
检测类别可在 detector.py 文件第60行修改。
修改测试视频目录default： parser.add_argument('--input_video_path', type=str, default='./video/test02.mp4',help='source video path.')  文件第262行
运行程序：python detect-car.py
"""
def main():
    # 根据视频尺寸，填充一个polygon，供撞线计算使用
    mask_image_temp = np.zeros((1080, 1920), dtype = np.uint8)

    # 初始化2个撞线polygon
    list_pts_blue = [[204, 305], [227, 431], [605, 522], [1101, 464], [1900, 601], [1902, 495], [1125, 379], [604, 437],
                     [299, 375], [267, 289]]
    ndarray_pts_blue = np.array(list_pts_blue, np.int32)
    polygon_blue_value_1 = cv2.fillPoly(mask_image_temp, [ndarray_pts_blue], color = 1)
    polygon_blue_value_1 = polygon_blue_value_1[:, :, np.newaxis]

    # 填充第二个polygon
    mask_image_temp = np.zeros((1080, 1920), dtype = np.uint8)
    list_pts_yellow = [[181, 305], [207, 442], [603, 544], [1107, 485], [1898, 625], [1893, 701], [1101, 568],
                       [594, 637], [118, 483], [109, 303]]
    ndarray_pts_yellow = np.array(list_pts_yellow, np.int32)
    polygon_yellow_value_2 = cv2.fillPoly(mask_image_temp, [ndarray_pts_yellow], color = 2)
    polygon_yellow_value_2 = polygon_yellow_value_2[:, :, np.newaxis]

    # 撞线检测用mask，包含2个polygon，（值范围 0、1、2），供撞线计算使用
    polygon_mask_blue_and_yellow = polygon_blue_value_1 + polygon_yellow_value_2

    # 缩小尺寸，1920x1080->960x540
    polygon_mask_blue_and_yellow = cv2.resize(polygon_mask_blue_and_yellow, (960, 540))

    # 蓝 色盘 b,g,r
    blue_color_plate = [255, 0, 0]
    # 蓝 polygon图片
    blue_image = np.array(polygon_blue_value_1 * blue_color_plate, np.uint8)

    # 黄 色盘
    yellow_color_plate = [0, 255, 255]
    # 黄 polygon图片
    yellow_image = np.array(polygon_yellow_value_2 * yellow_color_plate, np.uint8)

    # 彩色图片（值范围 0-255）
    color_polygons_image = blue_image + yellow_image
    # 缩小尺寸，1920x1080->960x540
    color_polygons_image = cv2.resize(color_polygons_image, (960, 540))

    # list 与蓝色polygon重叠
    list_overlapping_blue_polygon = []

    # list 与黄色polygon重叠
    list_overlapping_yellow_polygon = []

    # 进入数量
    down_count = 0
    # 离开数量
    up_count = 0

    font_draw_number = cv2.FONT_HERSHEY_SIMPLEX
    # draw_text_postion = (int(960 * 0.01), int(540 * 0.05))

    # 初始化 yolov5
    detector = Detector()
    opt=parse_opt()
    video_path =opt.input_video_path
    output_result_path = opt.output_result_path
    if not os.path.exists(opt.output_result_path):
        os.mkdir(opt.output_result_path)
    # capture = cv2.VideoCapture('/mnt/datasets/datasets/towncentre/TownCentreXVID.avi')
    video_file_name = video_path.split('/')[-1]  # 不带后缀的文件名
    video_file_name = video_file_name.split('.')[0]
    time_str = time.strftime('%m%d_%H%M', time.localtime())
    output_video_file_name = video_file_name + f'_{time_str}.mp4'
    output_video_path = os.path.join(output_result_path, output_video_file_name)
    # 全局变量
    global glob_file_path
    glob_file_path=os.path.join(os.getcwd(), output_video_path)

    # 打开视频
    capture = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # 获得码率及尺寸
    fps = capture.get(cv2.CAP_PROP_FPS)
    write_frame_size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    draw_text_postion = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) * 0.01), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) * 0.05))


    # video_out = cv2.VideoWriter(output_video_path, fourcc, fps, write_frame_size)  # 写入视频
    # 因为下方要对图片进行缩放，即全程整个图片大小为(960, 540)，不采用真实的write_frame_size，（1920,1080）
    video_out = cv2.VideoWriter(output_video_path, fourcc, fps, (960, 540))  # 写入视频

    while True:
        # 读取每帧图片
        _, im = capture.read()
        if im is None:
            break

        # 缩小尺寸，1920x1080->960x540
        im = cv2.resize(im, (960, 540))

        list_bboxs = []
        bboxes = detector.detect(im)

        # 如果画面中 有bbox
        if len(bboxes) > 0:
            list_bboxs = tracker.update(bboxes, im)

            # 画框
            # 撞线检测点，(x1，y1)，y方向偏移比例 0.0~1.0
            output_image_frame = tracker.draw_bboxes(im, list_bboxs, line_thickness = None)
            pass
        else:
            # 如果画面中 没有bbox
            output_image_frame = im
        pass

        # cv2.add图像运算方式需要输出的图像–必须与输入的图像具有相同的大小、类型和通道数
        # 查看通道数
        # print(output_image_frame.shape)
        # print(color_polygons_image.shape)
        # 输出图片，原始图片和长方形的图片进行拼接
        output_image_frame = cv2.add(output_image_frame, color_polygons_image)

        if len(list_bboxs) > 0:
            # ----------------------判断撞线----------------------
            for item_bbox in list_bboxs:
                x1, y1, x2, y2, label, track_id = item_bbox

                # 撞线检测点，(x1，y1)，y方向偏移比例 0.0~1.0
                y1_offset = int(y1 + ((y2 - y1) * 0.6))

                # 撞线的点
                y = y1_offset
                x = x1

                if polygon_mask_blue_and_yellow[y, x] == 1:
                    # 如果撞 蓝polygon
                    if track_id not in list_overlapping_blue_polygon:
                        list_overlapping_blue_polygon.append(track_id)
                    pass

                    # 判断 黄polygon list 里是否有此 track_id
                    # 有此 track_id，则 认为是 外出方向
                    if track_id in list_overlapping_yellow_polygon:
                        # 外出+1
                        up_count += 1

                        print(
                            f'类别: {label} | id: {track_id} | 上行撞线 | 上行撞线总数: {up_count} | 上行id列表: {list_overlapping_yellow_polygon}')

                        # 删除 黄polygon list 中的此id
                        list_overlapping_yellow_polygon.remove(track_id)

                        pass
                    else:
                        # 无此 track_id，不做其他操作
                        pass

                elif polygon_mask_blue_and_yellow[y, x] == 2:
                    # 如果撞 黄polygon
                    if track_id not in list_overlapping_yellow_polygon:
                        list_overlapping_yellow_polygon.append(track_id)
                    pass

                    # 判断 蓝polygon list 里是否有此 track_id
                    # 有此 track_id，则 认为是 进入方向
                    if track_id in list_overlapping_blue_polygon:
                        # 进入+1
                        down_count += 1

                        print(
                            f'类别: {label} | id: {track_id} | 下行撞线 | 下行撞线总数: {down_count} | 下行id列表: {list_overlapping_blue_polygon}')

                        # 删除 蓝polygon list 中的此id
                        list_overlapping_blue_polygon.remove(track_id)

                        pass
                    else:
                        # 无此 track_id，不做其他操作
                        pass
                    pass
                else:
                    pass
                pass

            pass

            # ----------------------清除无用id----------------------
            list_overlapping_all = list_overlapping_yellow_polygon + list_overlapping_blue_polygon
            for id1 in list_overlapping_all:
                is_found = False
                for _, _, _, _, _, bbox_id in list_bboxs:
                    if bbox_id == id1:
                        is_found = True
                        break
                    pass
                pass

                if not is_found:
                    # 如果没找到，删除id
                    if id1 in list_overlapping_yellow_polygon:
                        list_overlapping_yellow_polygon.remove(id1)
                    pass
                    if id1 in list_overlapping_blue_polygon:
                        list_overlapping_blue_polygon.remove(id1)
                    pass
                pass
            list_overlapping_all.clear()
            pass

            # 清空list
            list_bboxs.clear()

            pass
        else:
            # 如果图像中没有任何的bbox，则清空list
            list_overlapping_blue_polygon.clear()
            list_overlapping_yellow_polygon.clear()
            pass
        pass

        text_draw = 'DOWN: ' + str(down_count) + ' , UP: ' + str(up_count)

        # BGR颜色
        output_image_frame = cv2.putText(img = output_image_frame, text = text_draw,
                                         org = draw_text_postion,
                                         fontFace = font_draw_number,
                                         fontScale = 1, color = (0, 0, 255), thickness = 2)

        # 写入帧
        if video_out is not None:
            # 图片恢复大小
            # output_image_frame=cv2.resize(im, write_frame_size)
            video_out.write(output_image_frame)

        pass
    pass

    capture.release()
    video_out.release()


def parse_opt():
    parser = argparse.ArgumentParser()
    # person tracker params
    parser.add_argument('--input_video_path', type=str, default='./video/test.mp4',help='source video path.')
    parser.add_argument('--output_result_path', type=str, default='result/',help='output video inference result storage path.')
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    main()
