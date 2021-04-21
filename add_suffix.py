import os


def add_suffix(directory, suffix):
	for file in os.listdir(directory):
		new_name = f'{file.split(".")[0]}-{suffix}.png'
		os.rename(os.path.join(directory, file), os.path.join(directory, new_name))



directory = './out/cnn/train/synthetic/images'

add_suffix(directory, 22)
