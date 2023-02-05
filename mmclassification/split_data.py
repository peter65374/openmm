'''
# This tool is to pre-process the flower data to meet the requirement of classificaiton trainning, 
# following the openmm mmclassification0.25 training format.
'''

import os 
import sys 
import shutil
import numpy as np

def load_data(data_path):
    '''
    返回图片目录全部文件名：dict type
    {类别1: imgname list, 类别2: imgname list, 类别3: imgname list, ...}
    '''
    count = 0
    data = {}
    # 遍历数据目录下面的所有分类图片子目录
    for dir_name in os.listdir(data_path):
        dir_path = os.path.join(data_path, dir_name)
        if not os.path.isdir(dir_path):
            continue
        # 遍历单类图片目录下面的所有图片文件
        data[dir_name] = []
        for file_name in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file_name)
            if not os.path.isfile(file_path):
                continue
            data[dir_name].append(file_path)
        count += len(data[dir_name])
        print("image_num_{} :{}".format(dir_name, len(data[dir_name])))
    print("total of images : {}".format(count))
    return data


def copy_dataset(src_img_list, data_index, target_path): 
    '''
    按照data_index的顺序，把src_img_list的记录的文件数据copy到新的target_path.
    return: 返回目标文件路径的list
    '''
    target_img_list = []
    for index in data_index:
        src_img = src_img_list[index]
        img_name = os.path.split(src_img)[-1]
        shutil.copy(src_img, target_path)
        target_img_list.append(os.path.join(target_path, img_name))
    return target_img_list


def write_file(data, file_name):
    if isinstance(data, dict):
        write_data = []
        for lab, img_list in data.items():
            for img in img_list:
                write_data.append("{} {}".format(img, lab))
    else:
        write_data = data

    with open(file_name, "w") as f:
        for line in write_data:
            f.write(line + "\n")
    print("{} write over!".format(file_name))


def split_data(src_data_path, target_data_path, train_rate=0.8):
    '''按照train_rate的比例，随机shuffle次序并分割原始数据为train data和validation data'''
    src_data_dict = load_data(src_data_path)

    classes = []
    train_dataset, val_dataset = {}, {}
    train_count, val_count = 0, 0
    for i, (cls_name, img_list) in enumerate(src_data_dict.items()):
        # 按照train_rate, 随机排序分割成train data和validation data.
        img_data_size = len(img_list)
        random_index = np.random.choice(img_data_size, img_data_size,
                               replace=False)
        train_data_size = int(img_data_size * train_rate)
        train_data_index = random_index[:train_data_size]
        val_data_index = random_index[train_data_size:]
        # 按照mmclassification的数据格式要求，copy图像文件到对应路径
        train_data_path = os.path.join(target_data_path, "train", cls_name)
        val_data_path = os.path.join(target_data_path, "val", cls_name)
        os.makedirs(train_data_path, exist_ok=True)
        os.makedirs(val_data_path, exist_ok=True)
        classes.append(cls_name)
        train_dataset[i] = copy_dataset(img_list, train_data_index, train_data_path)
        val_dataset[i] = copy_dataset(img_list, val_data_index, val_data_path)
        print("target {} train:{}, val:{}".format(cls_name, len(train_dataset[i]), len(val_dataset[i])))
        train_count += len(train_dataset[i])
        val_count += len(val_dataset[i])
    print("train size:{}, val size:{}, total:{}".format(train_count, val_count, train_count + val_count))
    # 把数据文件的路径记录到相应txt文件中，以便training
    write_file(classes, os.path.join(target_data_path, "classes.txt"))
    write_file(train_dataset, os.path.join(target_data_path, "train.txt"))
    write_file(val_dataset, os.path.join(target_data_path, "val.txt"))


def main():
    src_data_path = sys.argv[1] # 源数据集路径
    target_data_path = sys.argv[2] # 处理后数据集目标路径
    split_data(src_data_path, target_data_path, train_rate=0.8)


# preprocessing command: python split_data.py [源数据集路径] [处理后数据集目标路径] 考虑到环境兼容建议使用绝对路径
if __name__ == '__main__': 
    main()
