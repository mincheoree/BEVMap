# 두개의 폴더에서 같은 이름을 갖는 이미지를 하나의 이미지로 합쳐서 저장하는 코드 생성
# 1. 이미지를 불러온다.
# 2. 이미지를 합친다.
# 3. 이미지를 저장한다.

import os
import cv2
import numpy as np

def combine_images(path1, path2, save_path):
    # 이미지를 불러온다.
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)
    # 이미지를 합친다.
    try:
        img = np.concatenate((img1, img2), axis=1)
    except:
        print(path1)
        return

    #path1의 Image는 baseline, paht2의 Image는 ours title 붙이기
    cv2.putText(img, 'Baseline', (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(img, 'Ours', (img1.shape[1], 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # 이미지를 저장한다.
    cv2.imwrite(save_path, img)


if __name__ == '__main__':
    # 이미지를 불러올 폴더
    path1 = '/root/Desktop/workspace/BEVDet/vis_baseline_trainval'
    path2 = '/root/Desktop/workspace/BEVDet/vis'

    # 이미지를 저장할 폴더
    save_path = '/root/Desktop/workspace/BEVDet/gather_vis'

    # 폴더 안의 이미지를 불러와서 합치고 저장한다.
    for file in os.listdir(path1):
        combine_images(os.path.join(path1, file), os.path.join(path2, file), os.path.join(save_path, file))