# import os

# label_dir = r'C:\Users\LOQ\Desktop\Infra_Damage_Detection\datasets\SDNET_YOLO\labels\train'

# files = os.listdir(label_dir)
# total = len(files)
# empty = 0
# bad_values = 0
# bad_files = []

# for i, fname in enumerate(files):
#     if i % 1000 == 0:
#         print(f'Progress: {i}/{total}...')
#     fpath = os.path.join(label_dir, fname)
#     with open(fpath) as f:
#         lines = f.readlines()
#     if len(lines) == 0:
#         empty += 1
#         continue
#     for line in lines:
#         parts = line.strip().split()
#         if len(parts) != 5:
#             bad_files.append((fname, 'wrong columns', line.strip()))
#             bad_values += 1
#             continue
#         cls, x, y, w, h = parts
#         if not (0 <= float(x) <= 1 and 0 <= float(y) <= 1 and 0 < float(w) <= 1 and 0 < float(h) <= 1):
#             bad_files.append((fname, 'out of range', line.strip()))
#             bad_values += 1

# print(f'Done!')
# print(f'Empty files: {empty}')
# print(f'Bad value lines: {bad_values}')
# print('Sample bad files:', bad_files[:10])


import cv2
import os
img_dir = r'C:\Users\LOQ\Desktop\Infra_Damage_Detection\datasets\SDNET_YOLO\images\train'
files = os.listdir(img_dir)
img = cv2.imread(os.path.join(img_dir, files[0]))
print('First image:', files[0])
print('Shape:', img.shape if img is not None else 'FAILED TO READ')