import os
import pickle
from dataHandler import FileName
from tqdm import  tqdm
import SimpleITK as sitk
import matplotlib.pyplot as plt
import vtk
import numpy as np
from vtk.util import numpy_support
from PIL import Image


def get_numpy_form_label(start_points, end_points):
    points = []
    for i in range(len(start_points)):
        s = start_points[i]
        e = end_points[i]
        points.append(s[0]), points.append(s[1])
        points.append(e[0]), points.append(e[1])
    return np.array(points, np.int32), len(points) / 4


def transform_to_fix_size(raw_image, height, width, start_points, end_points):
    raw_H, raw_W = raw_image.shape[0], raw_image.shape[1]
    im = Image.fromarray(raw_image)
    im = im.resize((width, height))

    rate_H, rate_W = height / raw_H, width / raw_W

    new_start_points = []
    new_end_points = []

    for s, e in zip(start_points, end_points):
        new_s = (s[0] * rate_W, s[1] * rate_H)
        new_e = (e[0] * rate_W, e[1] * rate_H)
        new_start_points.append(new_s)
        new_end_points.append(new_e)
    return np.array(im), new_start_points, new_end_points


def generateLabelPic(picsPath, outPutPath):
    os.makedirs(outPutPath, exist_ok=True)
    recordFileName = ""
    imagelist = []
    for _, _, filelist in os.walk(picsPath):
        for file in filelist:
            ext = os.path.splitext(file)[-1]
            if ext == ".dcm":
                imagelist.append(file)
            elif ext == ".ibrainRc":
                recordFileName = file

    imagelist.sort()
    record_dic = {}
    recordPath = "{}/{}".format(picsPath, recordFileName)
    with open(recordPath, 'rb') as f:
        record_dic = pickle.load(f)

    for index, name in tqdm(enumerate(imagelist)):
        flag = record_dic[name][FileName.Has_crack.Has_crack]
        if not flag:
            continue

        series = index
        imagePath = "{}/{}".format(picsPath, name)
        x = sitk.ReadImage(imagePath)
        xArray = sitk.GetArrayFromImage(x).squeeze()
        image_raw_data = xArray

        width = 488
        height = 488
        trans_data, new_start_points, new_end_points = transform_to_fix_size(image_raw_data, width, height,
                                                                             record_dic[name][
                                                                                 FileName.start_point.start_point],
                                                                             record_dic[name][FileName.end_point])
        line_label_array, pairNum = get_numpy_form_label(new_start_points, new_end_points)

        xs = line_label_array[0::2]
        ys = line_label_array[1::2]
        plt.imshow(trans_data, cmap='gray')
        plt.title("series:{}".format(series))
        plt.axis('off')
        plt.plot(xs, ys)
        savePath = "{}/{}.png".format(outPutPath, series)
        plt.savefig(savePath)
        plt.clf()

if __name__ == "__main__":
    root = "/home/zhenxingxu/teethData/pic{}"
    outputRoot = "./showLabel/pic{}"
    maxIndex = 37
    for index in range(1, maxIndex+1):
        print("handle pic{}".format(index))
        wkdir = root.format(index)
        output = outputRoot.format(index)
        generateLabelPic(wkdir, output)
