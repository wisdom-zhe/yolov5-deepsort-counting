import os
import time

import cv2
import numpy as np

glob_path='11'
def main():
	video_path = '/public/home/lgxy_20230991/zghuang/yolo-tracking/video/MOT16-13.mp4'
	output_result_path = 'result/'
	video_file_name = video_path.split('/')[-1]  # 不带后缀的文件名
	video_file_name = video_file_name.split('.')[0]
	time_str = time.strftime('%m%d_%H%M', time.localtime())
	output_video_file_name = video_file_name + f'_{time_str}.mp4'
	output_video_path = os.path.join(output_result_path, output_video_file_name)
	print(os.getcwd())
	download_file_path=os.path.join(os.getcwd(),output_video_path)

	print(output_video_file_name)
	print(output_video_path)

	print(download_file_path)
	global glob_path
	glob_path=download_file_path


def draw_line():
	# 背景图
	mask_image_temp = np.zeros((1080, 1920, 3), dtype = np.uint8)
	mask_image_temp2 = np.zeros((1080, 1920), dtype = np.uint8)

	# 初始化2个撞线polygon
	list_pts_blue = [[204, 305], [227, 431], [605, 522], [1101, 464], [1900, 601], [1902, 495], [1125, 379], [604, 437],
					 [299, 375], [267, 289]]
	ndarray_pts_blue = np.array(list_pts_blue, np.int32)
	polygon_blue_value_1 = cv2.fillPoly(mask_image_temp, [ndarray_pts_blue], color = (0, 255, 0))
	polygon_blue_value_2 = cv2.fillPoly(mask_image_temp2, [ndarray_pts_blue], color = (0, 255, 0))
	pass


if __name__ == '__main__':
	# main()
	# print(glob_path)
	draw_line()