import os
from os import path
import numpy as np
from sklearn.model_selection import train_test_split

root = './101_Categories'
categories = os.listdir(root)


for i in range(len(categories)):
    category = categories[i]
    print(category)
    cat_dir = path.join(root, category)

    images = os.listdir(cat_dir)

    images, images_test = train_test_split(images, test_size=0.15)
    images_train, images_val = train_test_split(images, test_size=0.2) #训练集：验证集：测试集=14:3:3
    image_sets = images_train, images_test, images_val
    labels = 'train', 'test', 'val'

    for image_set, label in zip(image_sets, labels):
        dst_folder = path.join(root, label, category)  # 创建文件夹
        os.makedirs(dst_folder)
            os.rename(src_dir, dst_dir)
    
    os.rmdir(cat_dir)  # 去除空文件夹
print(cnt1,cnt2,cnt3)