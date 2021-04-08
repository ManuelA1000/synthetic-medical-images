import os
import cv2


def clean_images(path):
	for root, dirs, files in os.walk(path):
	    for index, name in enumerate(files):
	        if name.endswith(('.png')):
	            img = cv2.imread(os.path.join(root, name))
	            img[img < 26] = 0
	            cv2.imwrite(os.path.join(root, name), img)
	            if index % 100 == 0:
	            	print(f'Files completed: {index + 1}')
	            	print(f'Current location: {root}')


clean_images(path = './out/gan/')
clean_images(path = './out/cnn/')
