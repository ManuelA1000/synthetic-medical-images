import os

import cv2
from tqdm import tqdm



IMAGE_EXTENSIONS = {'.bmp', '.jpg', '.jpeg','.png', '.tif', '.tiff'}

for root, dirs, files in os.walk('./data'):
    for name in tqdm(files):
        basename, ext = os.path.splitext(name)
        if ext.casefold() in IMAGE_EXTENSIONS:
            input_path = os.path.join(root, name)
            img = cv2.imread(input_path)
            img = cv2.resize(img, (256,256))
            img[img < 51] = 0
            save_path = os.path.join(root, basename + '.png')
            cv2.imwrite(save_path, img)
            if ext.casefold() != '.png':
                os.remove(input_path)
        else:
            pass
