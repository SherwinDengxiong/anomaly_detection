import cv2
import os
import torch
import math

from torch.autograd import Variable

import sys
from tqdm import tqdm

import numpy as np
from pytorch_i3d import InceptionI3d


def creatnetwork(mode, load_model):
    if mode == 'flow':
        i3d = InceptionI3d(400, in_channels=2)
    else:
        i3d = InceptionI3d(400, in_channels=3)
    # i3d.replace_logits(157)
    i3d.load_state_dict(torch.load(load_model))
    i3d.cuda()
    return i3d


def findsegment(frame_num, i, frameperseg, segnum):
    frame_num

    start = 0
    end = 0
    if (frame_num > segnum * frameperseg):
        perseg = math.floor(frame_num / segnum)
        if (i == segnum - 1):
            start = perseg * i
            end = frame_num
        else:
            start = perseg * i
            end = perseg * (i + 1)
    else:
        movementtem = math.ceil((segnum * frameperseg - frame_num) / (segnum - 1))
        movement = frameperseg - movementtem
        if (i == segnum - 1):
            start = (movement) * i
            end = frame_num
        else:
            start = (movement) * i
            end = start + frameperseg

    return start, end


def videotransform(image):
    image = np.transpose(image, [3, 0, 1, 2])
    image = np.expand_dims(image, axis=0)
    inputs = torch.from_numpy(image).to(torch.float32)
    return inputs


def readvideo(cap, frame_num, maxframenumber):
    image = []
    segmentindex = 1
    if frame_num > maxframenumber:
        segmentindex = math.ceil(frame_num / maxframenumber)
    else:
        segmentindex = 1

    for i in range(frame_num):
        ret, img = cap.read()
        if ret:
            img = cv2.resize(img, (224, 224))
            if i % segmentindex == 0:
                image.append(img)

    image = np.array(image)

    return image


def transformIntoFeature(inputs, i3d, cudaindex):
    b, c, t, h, w = inputs.shape

    # wrap them in Variable
    with torch.no_grad():
        inputs = Variable(inputs, volatile=True)

        features = i3d.extract_features(inputs.cuda(cudaindex))
        features = features.squeeze(0).permute(1, 2, 3, 0).data.cpu().numpy()
        features = np.squeeze(features)
    return features


def totalvideonumber(path):
    featurepath = path
    namelist = []
    number = 0
    for dir in sorted(os.listdir(featurepath)):
        dirpath = os.path.join(featurepath, dir)
        for video in sorted(os.listdir(dirpath)):
            videopath = os.path.join(dirpath, video)
            # print(videopath)
            number = number + 1
            namelist.append(videopath)
    return videopath, number


def deletedir(path):
    featurepath = path
    for dir in sorted(os.listdir(featurepath)):
        dirpath = os.path.join(featurepath, dir)
        for video in sorted(os.listdir(dirpath)):
            videopath = os.path.join(dirpath, video)
            videoseg = sorted(os.listdir(videopath))
            if len(videoseg) != 32:
                print(videopath)
                os.removedirs(videopath)


def testsingle():
    mode = "rgb"
    load_model = 'models/rgb_imagenet.pt'
    cudaindex = 0
    segnum = 1
    frameperseg = 10
    progress = 0
    maxsegment = 400 * segnum

    i3d = creatnetwork(mode, load_model)
    root=sys.argv[1]
    # root = '/media/sherwin/TOSHIBA EXT/CitySCENE/city_testing/cityscene_test/'

    featureroot = 'testing_feature_single'


    if not os.path.exists(featureroot + '/'):
        os.makedirs(featureroot)


    trainpath = root
    a = sorted(os.listdir(trainpath))
    for videoname in tqdm(a, total=len(a), desc="Extracting features"):
        videopath = os.path.join(trainpath, videoname)
        name = videoname[:-4]
        result_file = os.path.join(featureroot, name)
        if os.path.exists(result_file):
            continue

        cap = cv2.VideoCapture(videopath)

        videoori = readvideo(cap, int(cap.get(7)), maxsegment)
        framenum = videoori.shape[0]  # frame number
        if (framenum < (segnum + frameperseg)):
            for i in range(segnum):
                start = 0
                end = framenum
                inputtem = videoori[int(start):int(end)]
                inputs = videotransform(inputtem)
                features = transformIntoFeature(inputs, i3d, cudaindex)
                newfeature = torch.from_numpy(features)
                torch.save(newfeature, result_file)

                # np.save(os.path.join(featureroot, name), features)
            progress = progress + 1
            print(name)
            continue
        else:

            for i in range(segnum):
                start, end = findsegment(framenum, i, frameperseg, segnum)
                inputtem = videoori[int(start):int(end)]

                inputs = videotransform(inputtem)
                # print(name+"_"+str(i))

                features = transformIntoFeature(inputs, i3d, cudaindex)
                newfeature=torch.from_numpy(features)
                torch.save(newfeature, result_file)

                # np.save(os.path.join(featureroot, name+'.npy'), features)

            progress = progress + 1
            # print(str(progress / videonumber * 100) + '% complete. processing: ' + name)






if __name__ == '__main__':
    testsingle()
