import numpy as np
import os
img_list = []
path = '/home/aivs/바탕화면/hdd/adj_test/dataset/PNU_selected'
type_dir = os.listdir(path)
for dir in type_dir:
    type_path = os.path.join(path, dir)
    number_dir = os.listdir(type_path)
    for num in number_dir:
        number_path = os.path.join(type_path, num + '/capvideo')
        video_dir = os.listdir(number_path)
        for video in video_dir:
            video_path = os.path.join(number_path, video)
            frames = os.listdir(video_path)
            for frame in frames:
                img_path = os.path.join(video_path,frame)
                img_list.append(img_path)

img_list = np.sort(img_list)
path1 = '/home/aivs/바탕화면/hdd/adj_test/dataset/PNU_selected/img_list2.txt'
for path in img_list:
    with open(path1, 'a') as P:
        P.write(f'{path}\n')
