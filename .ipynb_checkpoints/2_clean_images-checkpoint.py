import os
import cv2


path = './out/gan/'

# for root, dirs, files in os.walk(path):
#     for name in files:
#         if name.endswith(('.png')):
#             img = cv2.imread(os.path.join(root, name))
#             img[img < 26] = 0
#             cv2.imwrite(os.path.join(root, name), img)


            
            
path = './out/cnn/'

for root, dirs, files in os.walk(path):
    for name in files:
        if name.endswith(('.png')):
            print(os.path.join(root, name))
#             img = cv2.imread(os.path.join(root, name))
#             img[img < 26] = 0
#             cv2.imwrite(os.path.join(root, name), img)