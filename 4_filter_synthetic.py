import os

import cv2


for root, dirs, files in os.walk('./out/train'):
    print(f'Currently in: {root}')
    for name in files:
        path = os.path.join(root, name)
        if path.endswith('.jpg'):
            new_path = os.path.splitext(path)[0] + '.png'
            img = cv2.imread(path)
            img[img < 51] = 0
            cv2.imwrite(new_path, img)
            os.remove(path)
