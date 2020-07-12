import numpy as np

import os

import random
random.seed(30)

from tqdm import tqdm

def calculategtpr(section,annomalysection):
    gt=0
    pr=0
    if section[0] != 0:
        if section[1]>annomalysection[0]:
            if section[1]>annomalysection[1]:
                gt=0
                pr=(section[2]-section[1])/(section[2]-section[1])+(annomalysection[1]-annomalysection[0])
            elif section[2]<annomalysection[1]:
                gt=(section[2]-section[1])/(annomalysection[1]-annomalysection[0])
                pr = (section[2] - section[1]) / (annomalysection[1] - annomalysection[0])
            elif section[2]>=annomalysection[1]:
                gt=(annomalysection[1]-section[1])/(section[2]-annomalysection[0])
                pr=(section[2]-section[1])/(section[2]-annomalysection[0])
        elif section[1]<=annomalysection[0]:
            if section[2]<annomalysection[0]:
                gt=0
                pr = (section[2] - section[1]) / (section[2] - section[1]) + (annomalysection[1] - annomalysection[0])
            elif section[2]>=annomalysection[1]:
                gt=(annomalysection[1] - annomalysection[0])/(section[2] - section[1])
                pr=1
            elif section[2]<annomalysection[1] and section[2]>=annomalysection[0]:
                gt=(section[2]-annomalysection[0])/(annomalysection[1]-section[1])
                pr=(section[2]-section[1])/(annomalysection[1]-section[1])
    elif section[0]==0:
        gt=0
        pr=0

    return np.array([int(section[0]),gt,pr,int(section[1]),int(section[2])])

def generateproposal(predict,frame,videoname,groundtruth):
    catagory=definecatagory(videoname)


    pro=[]
    probability=normalizepredict(predict)
    sum=0
    pro.append(sum)
    for i in range(len(probability)-1):
        sum=sum+probability[i]
        pro.append(sum)
    anomalysection = findanomalysection(predict,frame,groundtruth)
    repeatnumber = int(np.random.rand(1)*600)+200
    newsectionlist=[]
    for i in range(repeatnumber):
        section=generateOneproposal(predict,pro,frame,catagory,groundtruth)
        if section[2]-section[1]<=1:
            continue
        newsection= calculategtpr(section,anomalysection)
        newsectionlist.append(newsection)
    returnanomalysection=np.array([int(catagory),int(anomalysection[0]),int(anomalysection[1])])
    returnsection=np.array(newsectionlist)

    return returnanomalysection,returnsection

def findanomalysection(predict,frame,groundtruth):
    region=int(frame/len(predict))
    anomalypart=[]
    for i, score in enumerate(predict):
        if score>groundtruth:
            anomalypart.append(i*region)

    if len(anomalypart)==0:
        anomalypart.append(0)
        anomalypart.append(frame-1)

    anomalysection=np.array([anomalypart[0],anomalypart[-1]])
    return anomalysection

def generateOneproposal(predict,pro,frame,catagory,groundtruth):
    initrandomnumber1=np.random.rand(1)
    startframe=0
    for i,pronumber in enumerate(pro):
        if pronumber >initrandomnumber1:
            startframe=i
            break
        elif initrandomnumber1>pro[-1]:
            startframe=len(pro)-1
            break

    initrandomnumber2 = np.random.rand(1)
    endframe = 0
    for i, pronumber in enumerate(pro):
        if pronumber > initrandomnumber2:
            endframe = i
            break
        elif initrandomnumber2 > pro[-1]:
            endframe = len(pro) - 1
            break
    if startframe>endframe:
        temp=endframe
        endframe=startframe
        startframe=temp

    if np.mean(predict[startframe:endframe])<groundtruth:

        section=np.array([0,startframe,endframe])
    else:
        section=np.array([catagory,startframe,endframe])







    return section

def normalizepredict(predict):
    probability = []
    for i in range(len(predict)):
        probability.append(predict[i]/np.sum(predict))
    return probability
def generatepredict(catagoryname,scorelist,groundtruth):
    catagory=definecatagory(catagoryname)
    predict=[]
    for i,score in enumerate(scorelist):
        if score>=groundtruth:
            predict.append(catagory)
        else:
            predict.append(0)
    return np.array(predict)

def definecatagory(videoname):
    catagory = 0

    # normal
    if videoname.__contains__('normal'):
        catagory = 0
    elif videoname.__contains__('accident'):
        catagory = 1
    elif videoname.__contains__('carrying'):
        catagory = 2
    elif videoname.__contains__('crowd'):
        catagory = 3
    elif videoname.__contains__('explosion'):
        catagory = 4
    elif videoname.__contains__('fighting'):
        catagory = 5
    elif videoname.__contains__('graffiti'):
        catagory = 6
    elif videoname.__contains__('robbery'):
        catagory = 7
    elif videoname.__contains__('shooting'):
        catagory = 8
    elif videoname.__contains__('smoking'):
        catagory = 9
    elif videoname.__contains__('stealing'):
        catagory = 10
    elif videoname.__contains__('sweeping'):
        catagory = 11
    elif videoname.__contains__('walkingdog'):
        catagory = 12

    return catagory
def readfile(filepath):
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



def test_test():
    # select part of the training

    groundtruth = 0.5

    proposalfile='data/mil_test_result_proposal_list.txt'
    txttrain='result/task1'
    assert os.path.exists(txttrain), "Path does not exist! %s"%(txttrain)

    videopathlist=[]
    for i,ca in enumerate(sorted(os.listdir(txttrain))):

        capath=os.path.join(txttrain,ca)
        videopathlist.append(capath)
    videopathlist=np.array(videopathlist)





    #print(videopathlist.shape)
    onestring = ''
    stringlist = []

    for i, videopath in tqdm(enumerate(videopathlist),total=len(videopathlist)):
        catagoryname, filename,framenumber, scorelist=readfile(videopath)

        predictlist=scorelist



        onestring = onestring + '#' + str(i) + '\n'

        onestring = onestring + videopath[:-4] + '\n'

        onestring = onestring + str(framenumber) + '\n'
        onestring = onestring + str(1) + '\n'

        anomalysection, randomsection = generateproposal(predictlist, framenumber, videopath, groundtruth)
        onestring = onestring + str(1) + '\n'
        for index in anomalysection:
            onestring = onestring + str(int(index)) + ' '
        onestring = onestring + '\n'
        onestring = onestring + str(len(randomsection)) + '\n'
        for isection in randomsection:
            for j, number in enumerate(isection):
                if j == 1 or j == 2:
                    onestring = onestring + str(number) + ' '
                else:
                    onestring = onestring + str(int(number)) + ' '
            onestring = onestring + '\n'
        stringlist.append(onestring)
        onestring = ''

    file = open(proposalfile, 'w')
    for astring in stringlist:
        file.write(astring)
    file.close()
if __name__ == '__main__':
    test_test()
