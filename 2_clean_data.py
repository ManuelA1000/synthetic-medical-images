import os
import cv2


def clean_images(path):
	for root, dirs, files in os.walk(path):
		print(f'Current location: {root}')
		for index, name in enumerate(files):
			if name.endswith(('.bmp')):
				png_name = name.split('.')[0] + '.png'
				img = cv2.imread(os.path.join(root, name))
				img[img < 26] = 0
				cv2.imwrite(os.path.join(root, png_name), img)
				os.remove(os.path.join(root, name))
					


clean_images(path = './out/')
