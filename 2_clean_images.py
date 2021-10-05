import os

import cv2
from tqdm import tqdm

for root, dirs, files in os.walk('./out'):
   for name in tqdm(files):
       if name.endswith('.bmp'):
           input_path = os.path.join(root, name)
           img = cv2.imread(input_path)
           img[img < 26] = 0
           save_path = os.path.splitext(input_path)[0] + '.png'
           cv2.imwrite(save_path, img)
           os.remove(input_path)
       else:
           pass
