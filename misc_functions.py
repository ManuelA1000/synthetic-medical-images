import os
import shutil
import cv2



def create_copies(path, num_copies):
	store_dir = os.path.join(path, '../copied')
	os.mkdir(store_dir)
	for copy_num in range(num_copies):
		for file in os.listdir(path):
			new_name = f'{file.split(".")[0]}-{copy_num + 1}.bmp'
			shutil.copyfile(os.path.join(path, file), os.path.join(store_dir, new_name))


def clean_bmp(path):
	for root, dirs, files in os.walk(path):
		print(f'Current location: {root}')
		for index, name in enumerate(files):
			if name.endswith(('.bmp')):
				png_name = name.split('.')[0] + '.png'
				img = cv2.imread(os.path.join(root, name))
				img[img < 26] = 0
				cv2.imwrite(os.path.join(root, png_name), img)
				os.remove(os.path.join(root, name))


def clean_png(path):
	for root, dirs, files in os.walk(path):
		print(f'Current location: {root}')
		for index, name in enumerate(files):
			if name.endswith(('.png')):
				img = cv2.imread(os.path.join(root, name))
				img[img < 26] = 0
				cv2.imwrite(os.path.join(root, name), img)


# create_copies('./out/cnn/train/real/2/', 6)

# create_copies('./out/cnn/train/real/3/', 24)

# clean_bmp('./out/')

# clean_png('./out/cnn/train/synthetic/')

# clean_png('./out/cnn/test/set_100/')
