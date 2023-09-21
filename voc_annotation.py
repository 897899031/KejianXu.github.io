import os
import random
import xml.etree.ElementTree as ET
import numpy as np
from utils.util import get_classes

# --------------------------------------------------------------------------------------------------------------------------------#
# annotation_mode is used to specify the content of the file runtime calculations
# annotation_mode is 0 for the whole labeling process, including obtaining the txt inside VOCdevkit/VOC2007/ImageSets and 2007_train.txt and 2007_val.txt for training.
# annotation_mode is 1 for getting the txt in VOCdevkit/VOC2007/ImageSets.
# annotation_mode 2 means get 2007_train.txt, 2007_val.txt for training.
# --------------------------------------------------------------------------------------------------------------------------------#
annotation_mode = 0
# -------------------------------------------------------------------#
# Must be modified for generating target information for 2007_train.txt, 2007_val.txt
# Just be consistent with the classes_path used for training and prediction.
# If there is no target information in the generated 2007_train.txt
# If the generated 2007_train.txt doesn't have target information, then it's because the classes aren't set correctly.
# Only works when annotation_mode is 0 and 2.
# -------------------------------------------------------------------#
classes_path = 'model_data/voc_classes.txt'
# --------------------------------------------------------------------------------------------------------------------------------#
# trainval_percent is used to specify the ratio of (training set + validation set) to test set, by default (training set + validation set):test set = 9:1
# train_percent is used to specify the ratio of training set to validation set in (training set + validation set), by default train:validation set = 9:1
# Only valid when annotation_mode is 0 and 1.
# --------------------------------------------------------------------------------------------------------------------------------#
trainval_percent = 0.9
train_percent = 0.9
# -------------------------------------------------------#
# Point to the folder where the VOC dataset is located
# Point to the VOC dataset in the root directory by default
# -------------------------------------------------------#
VOCdevkit_path = 'VOCdevkit'

VOCdevkit_sets = [('2012', 'train'), ('2012', 'val')]
classes, _ = get_classes(classes_path)

# -------------------------------------------------------#
# statistical targets
# -------------------------------------------------------#
photo_nums = np.zeros(len(VOCdevkit_sets))
nums = np.zeros(len(classes))


def convert_annotation(year, image_id, list_file):
    in_file = open(os.path.join(VOCdevkit_path, 'VOC%s/Annotations/%s.xml' % (year, image_id)), encoding='utf-8')
    tree = ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = 0
        if obj.find('difficult') != None:
            difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('ymin').text)),
             int(float(xmlbox.find('xmax').text)), int(float(xmlbox.find('ymax').text)))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))

        nums[classes.index(cls)] = nums[classes.index(cls)] + 1


if __name__ == "__main__":
    random.seed(0)
    if " " in os.path.abspath(VOCdevkit_path):
        raise ValueError("The path of the folder where the dataset is stored and the name of the image must not contain spaces, otherwise it will affect the normal model training, please pay attention to modify。")

    if annotation_mode == 0 or annotation_mode == 1:
        print("Generate txt in ImageSets.")
        xmlfilepath = os.path.join(VOCdevkit_path, 'VOC2012/Annotations')
        saveBasePath = os.path.join(VOCdevkit_path, 'VOC2012/ImageSets/Main')
        temp_xml = os.listdir(xmlfilepath)
        total_xml = []
        for xml in temp_xml:
            if xml.endswith(".xml"):
                total_xml.append(xml)

        num = len(total_xml)
        list = range(num)
        tv = int(num * trainval_percent)
        tr = int(tv * train_percent)
        trainval = random.sample(list, tv)
        train = random.sample(trainval, tr)

        print("train and val size", tv)
        print("train size", tr)
        ftrainval = open(os.path.join(saveBasePath, 'trainval.txt'), 'w')
        ftest = open(os.path.join(saveBasePath, 'test.txt'), 'w')
        ftrain = open(os.path.join(saveBasePath, 'train.txt'), 'w')
        fval = open(os.path.join(saveBasePath, 'val.txt'), 'w')

        for i in list:
            name = total_xml[i][:-4] + '\n'
            if i in trainval:
                ftrainval.write(name)
                if i in train:
                    ftrain.write(name)
                else:
                    fval.write(name)
            else:
                ftest.write(name)

        ftrainval.close()
        ftrain.close()
        fval.close()
        ftest.close()
        print("Generate txt in ImageSets done.")

    if annotation_mode == 0 or annotation_mode == 2:
        print("Generate 2012_train.txt and 2012_val.txt for train.")
        type_index = 0
        for year, image_set in VOCdevkit_sets:
            image_ids = open(os.path.join(VOCdevkit_path, 'VOC%s/ImageSets/Main/%s.txt' % (year, image_set)),
                             encoding='utf-8').read().strip().split()
            list_file = open('%s_%s.txt' % (year, image_set), 'w', encoding='utf-8')
            for image_id in image_ids:
                list_file.write('%s/VOC%s/JPEGImages/%s.jpg' % (os.path.abspath(VOCdevkit_path), year, image_id))

                convert_annotation(year, image_id, list_file)
                list_file.write('\n')
            photo_nums[type_index] = len(image_ids)
            type_index += 1
            list_file.close()
        print("Generate 2012_train.txt and 2012_val.txt for train done.")


        def printTable(List1, List2):
            for i in range(len(List1[0])):
                print("|", end=' ')
                for j in range(len(List1)):
                    print(List1[j][i].rjust(int(List2[j])), end=' ')
                    print("|", end=' ')
                print()


        str_nums = [str(int(x)) for x in nums]
        tableData = [
            classes, str_nums
        ]
        colWidths = [0] * len(tableData)
        len1 = 0
        for i in range(len(tableData)):
            for j in range(len(tableData[i])):
                if len(tableData[i][j]) > colWidths[i]:
                    colWidths[i] = len(tableData[i][j])
        printTable(tableData, colWidths)

        if photo_nums[0] <= 500:
            print("The number of training sets is less than 500, which is a small amount of data, so please take care to set a larger training generation (Epoch) to meet the sufficient number of gradient descents (Step)")

        if np.sum(nums) == 0:
            print(
                "No targets were obtained in the dataset, please take care to modify the classes_path to correspond to your own dataset and make sure the label names are correct, otherwise the training will be ineffective!")
