import numpy as np
import cv2
import os
os.environ['KERAS_BACKEND'] = 'theano'
import keras.backend as K 
K.set_image_dim_ordering('th')
import random
random.seed(30)

from keras.models import Sequential,load_model
import sys
from scipy.io import loadmat, savemat
from keras.models import model_from_json

def load_model(json_path):  # Function to load the model
    model = model_from_json(open(json_path).read())
    return model
def conv_dict(dict2):
    i = 0
    dict = {}
    for i in range(len(dict2)):
        if str(i) in dict2:
            if dict2[str(i)].shape == (0, 0):
                dict[str(i)] = dict2[str(i)]
            else:
                weights = dict2[str(i)][0]
                weights2 = []
                for weight in weights:
                    if weight.shape in [(1, x) for x in range(0, 5000)]:
                        weights2.append(weight[0])
                    else:
                        weights2.append(weight)
                dict[str(i)] = weights2
    return dict
def load_weights(model, weight_path):  # Function to load the model weights
    dict2 = loadmat(weight_path)
    dict = conv_dict(dict2)
    i = 0
    for layer in model.layers:
        weights = dict[str(i)]
        layer.set_weights(weights)
        i += 1
    return model
def getvideoinput(videopath):
    videofeature=[]
    for featurename in sorted(os.listdir(videopath)):
        if type(featurename) is bytes:
            
            feature=np.load(os.path.join(videopath,featurename.decode("utf-8")))
        else:
            feature=np.load(os.path.join(videopath,featurename))
        if feature.shape==(1024,):
            tempfeature=feature.T.tolist()
            videofeature.append(tempfeature)
        else:
            # tempfeature=np.mean(feature,axis=0).T.tolist()
            # videofeature.append(tempfeature)
            for i,tempfeature in enumerate(feature):
                videofeature.append(tempfeature.T.tolist())
    return np.array(videofeature)


def loadinputs(trainvideopath):
    allvideopath=[]
    for dataset in os.listdir(trainvideopath):
        testdatasetpath=os.path.join(trainvideopath,dataset)
        for testvideoname in sorted(os.listdir(testdatasetpath)):
            testvideopath=os.path.join(testdatasetpath,testvideoname)
            allvideopath.append(testvideopath)
            # videofeature=getvideoinput(testvideopath)
            # for i, line in enumerate(videofeature):
            #     allfeature.append(line)

    return np.array(allvideopath)
def videonametransformtrain(featurepath,videopath):
    flist=featurepath.split('/')
    videoname=flist[-1]
    videopath=os.path.join(videopath,flist[-2],videoname)

    return videopath
def videonametransform(featurepath,videopath):
    flist=featurepath.split('/')
    videoname=flist[-1]
    videopath=os.path.join(videopath,videoname)

    return videopath
def readvideo(videopath):
    cap = cv2.VideoCapture(videopath)

    return cap.get(7)
def generatepredicttxt(predict,videoframenum,seg,framepreseg):
    ratio=1.0
    if videoframenum<(seg+framepreseg):
        trueseg=int(len(predict)/seg)
        ratio=trueseg/videoframenum

    else:
        trueseg=len(predict)
        ratio = trueseg / videoframenum

    framepredict=[]
    for i in range(int(videoframenum)):
        segnum=int(i*ratio)
        line=predict[segnum]
        framepredict.append(line)

    return framepredict



def test2():
    trainvideopath='testing_feature_multi/'
    videoroot= sys.argv[1]
    #videoroot = '/media/sherwin/TOSHIBA EXT/CitySCENE/city_testing/cityscene_test/'
    txtroot='result/task1'

    testpath = 'test3'
    percent = 'percent9'

    modelname = 'finaltest1'
    seg=32
    framepreseg=10

    percentpath = os.path.join(testpath, percent)
    mpath=os.path.join(percentpath,modelname)
    model_path = mpath + '.json'
    weights_path = mpath + '.mat'

    model = load_model(model_path)
    load_weights(model, weights_path)

    inputspath=loadinputs(trainvideopath)
    if not os.path.exists(txtroot):
        os.makedirs(txtroot)
    for i, featurepath in enumerate(inputspath):
        featurelistinput=getvideoinput(featurepath)
        videopredict=model.predict_on_batch(featurelistinput)
        predict=(1-videopredict).T[0]
        video=videonametransform(featurepath,videoroot)
        vname=video.split('/')[-1]
        videoframenum=readvideo(video+'.mp4')
        framepredict=generatepredicttxt(predict,videoframenum,seg,framepreseg)
        strings='video <'+vname+'>\n'
        for i,score in enumerate(framepredict):
            number=('%.3f'%score)
            strings+=str(i)+' <'+str(number)+'>\n'

        txtname=os.path.join(txtroot,vname+'.txt')
        with open(txtname,'w') as f:
            f.write(strings)
        count=0



if __name__ == '__main__':
    test2()
