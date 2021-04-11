import scipy.io as sio
import pandas as pd
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
import sys
import cv2
from moviepy.editor import *
import numpy as np
import argparse


# 数据集中‘Pose_Para’表示姿态方位
# 分别表示pitch、yaw、roll、tdx、tdy、tdz、scale_factor
# tdx、tdy和tdz为平移矩阵的数据，scale_factor表示相机变换的尺度系数

def get_args():
	parser = argparse.ArgumentParser(description="This script cleans-up noisy labels "
	                                             "and creates database for training.",
	                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument("--db", type=str, default='./AFW',
	                    help="path to database")
	parser.add_argument("--output", type=str, default='./AFW.npz',
	                    help="path to output database mat file")
	parser.add_argument("--img_size", type=int, default=64,
	                    help="output image size")
	parser.add_argument("--ad", type=float, default=0.6,
	                    help="enlarge margin")


	args = parser.parse_args()
	return args


def main():
	args = get_args()
	mypath = args.db
	output_path = args.output
	img_size = args.img_size
	ad = args.ad

	isPlot = True

	# 获取mypath路径下所有的.mat文件名称
	onlyfiles_mat = [f for f in listdir(mypath) if isfile(join(mypath, f)) and join(mypath, f).endswith('.mat')]
	# 获取mypath路径下所有的.jpg文件名称
	onlyfiles_jpg = [f for f in listdir(mypath) if isfile(join(mypath, f)) and join(mypath, f).endswith('.jpg')]
	# 进行排序
	onlyfiles_mat.sort()
	onlyfiles_jpg.sort()

	print(len(onlyfiles_jpg))
	print(len(onlyfiles_mat))
	# 保存所有图像像素
	out_imgs = []
	# 保存所有图像对应的姿态
	out_poses = []

	for i in tqdm(range(len(onlyfiles_jpg))):
		# 获得图片路径
		img_name = onlyfiles_jpg[i]
		# 获得图片对应的mat文件路径
		mat_name = onlyfiles_mat[i]

		img_name_split = img_name.split('.')
		mat_name_split = mat_name.split('.')

		if img_name_split[0] != mat_name_split[0]:
			print('Mismatched!')
			sys.exit()

		# 加载mat文件
		mat_contents = sio.loadmat(mypath + '/' + mat_name)
		pose_para = mat_contents['Pose_Para'][0]

		# 获得人脸的68个关键点坐标
		pt2d = mat_contents['pt2d']
		pt2d_x = pt2d[0,:]
		pt2d_y = pt2d[1,:]

		# I found negative value in AFLW2000. It need to be removed.
		pt2d_idx = pt2d_x>0.0
		pt2d_idy= pt2d_y>0.0

		pt2d_id = pt2d_idx
		if sum(pt2d_idx) > sum(pt2d_idy):
			pt2d_id = pt2d_idy
		# 保存68个关键点的x，y坐标（坐标如果小于0，则已经被剔除）
		pt2d_x = pt2d_x[pt2d_id]
		pt2d_y = pt2d_y[pt2d_id]
		
		img = cv2.imread(mypath+'/'+img_name)
		img_h = img.shape[0]
		img_w = img.shape[1]

		# 可以取消注释，查看68个关键点
		# point_size = 1
		# point_color = (0, 0, 255)  # BGR
		# thickness = 4  # 可以为 0 、4、8
		# poins = zip(pt2d_x, pt2d_y)
		# for x,y in poins:
		# 	cv2.circle(img, (int(x),int(y)), point_size, point_color, thickness)
		# cv2.imshow('check', img)
		# k = cv2.waitKey(500)

		# 根据关键点，对图片进行剪裁
		# Crop the face loosely
		x_min = int(min(pt2d_x))
		x_max = int(max(pt2d_x))
		y_min = int(min(pt2d_y))
		y_max = int(max(pt2d_y))
		
		h = y_max-y_min
		w = x_max-x_min

		# ad = 0.4
		# ad*w表示，根据关键点剪切的时候，其增加剪切边缘的大小
		x_min = max(int(x_min - ad * w), 0)
		x_max = min(int(x_max + ad * w), img_w - 1)
		y_min = max(int(y_min - ad * h), 0)
		y_max = min(int(y_max + ad * h), img_h - 1)
		
		img = img[y_min:y_max,x_min:x_max]
		# 查看剪切之后的效果
		if isPlot:
			cv2.imshow('check',img)
			k=cv2.waitKey(500)

		img = cv2.resize(img, (img_size, img_size))

		# 获得对应三维空间的旋转角度
		pitch = pose_para[0] * 180 / np.pi
		yaw = pose_para[1] * 180 / np.pi
		roll = pose_para[2] * 180 / np.pi

		# 把旋转角度当做标签，加入数组
		cont_labels = np.array([yaw, pitch, roll])

		out_imgs.append(img)
		out_poses.append(cont_labels)

	np.savez(output_path,image=np.array(out_imgs), pose=np.array(out_poses), img_size=img_size)


if __name__ == '__main__':
	main()