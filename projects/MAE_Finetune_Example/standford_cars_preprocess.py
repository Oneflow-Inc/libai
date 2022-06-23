import os
import os.path as osp
import shutil
import argparse

import numpy as np
import cv2
from tqdm import tqdm
import scipy.io as scio
import matplotlib.pyplot as plt


def h_w_a_statistics(data_dir):
    """ Make statistics about the height, width and area of images.
    """
    max_h, max_w = 0, 0
    max_area = 0
    heights, widths, areas = [], [], []
    for filename in tqdm(os.listdir(data_dir)):
        h, w, _ = cv2.imread(osp.join(data_dir, filename)).shape
        if h > max_h:
            max_h = h
        if w > max_w:
            max_w = w
        
        area = h * w
        if area > max_area:
            max_area = area
        
        heights.append(h)
        widths.append(w)
        areas.append(area)

    print(max_h, max_w, max_area)

    plt.subplot(311)
    plt.hist(heights, bins=20)
    plt.subplot(312)
    plt.hist(widths, bins=20)
    plt.subplot(313)
    plt.hist(areas, bins=20)
    plt.savefig('./statistics.jpg')


def equal_proportion_scaling(src_dir, dst_dir, dst_hw=(224, 224)):
    """ Offline scaling images, padding with 0.
    """
    if not osp.exists(dst_dir):
        os.mkdir(dst_dir)

    for filename in tqdm(os.listdir(src_dir)):
        img = cv2.imread(osp.join(src_dir, filename))
        
        h, w, _ = img.shape
        # use the longer side to determine scale multiple
        scale = dst_hw[1] / w if w >= h else dst_hw[0] / h
        
        scaled_img = np.zeros((*dst_hw, 3), dtype=np.uint8)
        img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
        scaled_img[:img.shape[0], :img.shape[1]] = img
        
        cv2.imwrite(osp.join(dst_dir, filename), scaled_img)


def split_into_train_test(data_root, img_dir, anno_path):
    """ Split the data into train set and test set. 
    """
    data = scio.loadmat(anno_path)  # read the .mat file
    images = data['annotations'][0]

    class_names = [str(class_name[0]) for class_name in data['class_names'][0]]
    # Special process for the class name containing '/', which cannot exist in file paths
    class_names[173] = class_names[173].replace('/', '-')

    train_dir = osp.join(data_root, 'train')
    if not osp.exists(train_dir):
        os.mkdir(train_dir)
    test_dir = osp.join(data_root, 'test')
    if not osp.exists(test_dir):
        os.mkdir(test_dir)

    for class_name in class_names:
        os.mkdir(osp.join(train_dir, class_name))
        os.mkdir(osp.join(test_dir, class_name))

    for i in range(images.size):
        filename = images[i][0][0]  # 'car_ims/xxx.jpg'
        filename = filename.split('/')[1]   
        class_id = images[i][5][0][0]
        class_id = class_id.astype(np.int32)
        class_name = class_names[class_id-1]
        is_test = images[i][6][0]  # train / test
        if is_test:
            shutil.copyfile(osp.join(img_dir, filename), osp.join(test_dir, class_name, filename))
        else:
            shutil.copyfile(osp.join(img_dir, filename), osp.join(train_dir, class_name, filename))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=str, required=True, help='The root directory of the dataset')
    args = parser.parse_args()

    img_dir = osp.join(args.dataset_root, 'car_ims')
    temp_scaled_path = osp.join(args.dataset_root, 'scaled_224')
    equal_proportion_scaling(img_dir, temp_scaled_path)
    split_into_train_test(args.dataset_root, temp_scaled_path, osp.join(args.dataset_root, 'cars_annos.mat'))
