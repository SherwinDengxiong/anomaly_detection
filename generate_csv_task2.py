
import pickle
import pandas as pd
import numpy as np

import os
import random

import math

from tqdm import tqdm
def findanomalysection(filename,framenumber,allanomalysection):
    anomalylist=[]
    clustername=['Normal','Accident','Carrying','Crowd','Explosion','Fighting','Graffiti','Robbery','Shooting','Smoking','Stealing','Sweeping','Walkingdog']
    for i,anomalysection in enumerate(allanomalysection):
        if filename==anomalysection[0]:
            anomalylist.append(anomalysection)
    if len(anomalylist)==1:
        catagory=anomalylist[0][1]
        startframe=math.floor(anomalylist[0][2]*(framenumber-1))
        endframe = math.floor(anomalylist[0][3] * (framenumber - 1))
        return [filename,clustername[catagory],str(startframe),str(endframe)]
    elif len(anomalylist)>1:
        anomalylist=sorted(anomalylist, key=(lambda x:x[4]) )
        catagory = anomalylist[-1][1]
        startframe = math.floor(anomalylist[-1][2] * (framenumber - 1))
        endframe = math.floor(anomalylist[-1][3] * (framenumber - 1))
        return [filename,clustername[catagory],str(startframe),str(endframe)]
    elif len(anomalylist)==0:
        catagory=random.randint(1,12)
        startframe=0
        endframe=framenumber-1
        return [filename,clustername[catagory],str(startframe),str(endframe)]

def isanomaly(scorelist,groundtruth,anomalyframenumber):
    count=0
    for score in scorelist:
        count+=score
    if (count/len(scorelist))>groundtruth:
        return True
    else:
        return False
def selectsection(result,groundtruth):
    anomalyvideosection=[]
    for i, videosection in enumerate(result):
        if videosection[-1]>groundtruth:
            temparray=[videosection[0],videosection[1]+1,videosection[2],videosection[3],videosection[4]]
            anomalyvideosection.append(temparray)
    return anomalyvideosection
def gtcontain(gt,anomalysection):
    videolist=[]
    T=0
    for i,videosection in enumerate(anomalysection):

        videolist.append(videosection[0])
    for i,gtvideo in enumerate(gt):
        if videolist.__contains__(gtvideo[0]):
            T+=1

    return T,len(gt),len(anomalysection)
def readtxtfile(filepath):
    catagoryname=''
    filename=''
    scorelist=[]
    count = 0
    with open(filepath,'r') as f:

        for i, str in enumerate(f):
            if i==0:
                file=str.split(' ')[1]
                filename=file[1:-2]
                catagoryname=filename.split('_')[0]

            else:
                line=str.split(' ')[1]
                score=line[1:-3]
                scorelist.append(float(score))
                count+=1

    return [catagoryname,filename,count,np.array(scorelist)]
def readfile(filename):
    pk = open(filename, 'rb')
    pred = pickle.load(pk)
    return pred





def loadcsvresult():
    filename = 'pred_dump_result.pc'
    txttrain = 'result/task1'
    csvroot='result/task2'
    cfilename='predictions.csv'
    pred = readfile(filename)

    groundtruthscore = 14
    milgrounttruth=0.6
    anomalyframenumber=5

    if not os.path.exists(csvroot):
        os.makedirs(csvroot)
    csvfilename=os.path.join(csvroot,cfilename)
    allanomalysection=[]
    for i, cluster in enumerate(pred):
        result = pd.DataFrame(cluster).to_numpy()
        anomalysection = selectsection(result, groundtruthscore)
        for section in anomalysection:
            allanomalysection.append(section)
    allanomalysection=sorted(allanomalysection,key=(lambda x:x[0]))
    videopathlist=[]
    for i,ca in enumerate(sorted(os.listdir(txttrain))):

        capath=os.path.join(txttrain,ca)
        videopathlist.append(capath)
    videopathlist=np.array(videopathlist)

    csvlist=[]
    for i, videopath in tqdm(enumerate(videopathlist),total=len(videopathlist)):
        catagoryname, filename,framenumber, scorelist=readtxtfile(videopath)
        if isanomaly(scorelist,milgrounttruth,anomalyframenumber):
            anomalytemp=findanomalysection(filename,framenumber,allanomalysection)
            csvlist.append(anomalytemp)
        else:
            anomalytemp=[filename,'Normal','-1','-1']
            csvlist.append(anomalytemp)
    csvlist=np.array(csvlist)
    np.savetxt(csvfilename,csvlist,delimiter=",",fmt="%s,%s,%s,%s")





if __name__ == '__main__':
    loadcsvresult()
