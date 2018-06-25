import cv2 as cv
import numpy as np
from Misc import *

def testImages():
    path = 'D:\CVis\\000000_10.png'
    path2 = 'D:\CVis\\000000_11.png'
    return cv.imread(path), cv.imread(path2)

def testVideo():
    path = 'D:\CVis\\driver.mp4'
    return cv.VideoCapture(path)

def imageWork():
    images = []
    OriginalImage, OriginalImage2 = testImages()
    images.append(OriginalImage)
    img = PreWork(OriginalImage)
    img = TestWork(img)
    images.append(img)
    #img = liner(img)
    #showMe(img,0)
    images.append(img)
    result1 = blend_transparent(OriginalImage,img)
    images.append(result1)
    img,sgs = edge2Segmented(img)
    img = SegImg(sgs)
    img = cv.applyColorMap(img,cv.COLORMAP_HSV)
    #showMe(img,0)
    images.append(img)
    sg = sgs[1]*255
    img,rect = GetRect(sg)
    img,rRect = GetRRect(sg)
    images.append(img)
    img = movement(OriginalImage,OriginalImage2,rect)
    images.append(overlayRRect(OriginalImage,rRect))
    images.append(img)
    showMany(images)

def videoWork():
    cap = testVideo()
    for i in range(1000):
        ret, frame = cap.read()
    
    while(cap.isOpened()):
        ret, frame = cap.read()
        ret, frame = cap.read()
        ret, frame = cap.read()
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv.resize(frame,(0,0), fx = 0.50,fy =0.50)
        count = 0;
        img = frame
        img = PreWork(img)
        img = TestWork(img)
        img,sgs = edge2Segmented(img)
        img = SegImg(sgs)
        #img = cv.resize(img,(0,0), fx = 2,fy =2)
        #frame = cv.resize(frame,(0,0), fx = 2,fy =2)
        img = cv.applyColorMap(img,cv.COLORMAP_HSV)
        img = blend_transparent(frame,img)

        cv.destroyAllWindows()
        showMe(img,1)

    cap.release()
    cv.destroyAllWindows()
    return;

def PreWork(img):
    img = blur(img,5)
    img = OpenClose(img,5)
    img = multMed(img, 1,5)
    img = MeanS(img,20)
    return img

def TestWork(img):
    ced = CEdged(img,100)
    #hed = HEdged(img, 50)
    img = ced#|hed
    img = closer(img,5)
    img = Skeleton(img,3)
    img = closePix2(img)
    img = closer(img,10)
    img = CEdged(img, 100)
    img = closer(img,5)
    img = Skeleton(img,3)
    img = closePix2(img)
    return img



def contourize(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(gray, 127, 255, 0)
    im2, contours, hierarchy = cv.findContours(np.int32(thresh), cv.RETR_FLOODFILL, cv.CHAIN_APPROX_SIMPLE)
    img = G2R(np.uint8(im2))
    return img, contours

def movement(img,img2, rect):
    roi = getROI(img,rect)
    window = (rect)
    rect2 = (rect[0] + int(rect[2]/2), rect[1] + int(rect[3]/2), rect[2]*4, rect[3]*4)
    windowC = (rect)

    x,y,w,h = rect
    img3 = cv.rectangle(img2,(x,y),(x+w,y+h),(0,255,0),2)
    #showMe(img3,0)


    roiHsv = cv.cvtColor(roi, cv.COLOR_BGR2HSV)

    lowLimit = np.array((0., 60., 32.))
    highLimit = np.array((180., 255., 255.))
    mask = cv.inRange(roiHsv, lowLimit, highLimit)
    roiHist = cv.calcHist([roiHsv], [0], mask, [180], [0, 180])
    cv.normalize(roiHist, roiHist, 0, 255, cv.NORM_MINMAX)

    
    terminationCriteria = (cv.TERM_CRITERIA_COUNT | cv.TERM_CRITERIA_EPS , 50, 500)

    #frameHsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    #backprojectedFrame = cv.calcBackProject([frameHsv], [0], roiHist, [0, 180], 1)

    #mask = cv.inRange(frameHsv, lowLimit, highLimit)
    #backprojectedFrame &= mask

    #ret, window = cv.CamShift(backprojectedFrame, window, terminationCriteria)





    frameHsv = cv.cvtColor(img2, cv.COLOR_BGR2HSV)
    backprojectedFrame = cv.calcBackProject([frameHsv], [0], roiHist, [0, 180], 1)

    mask = cv.inRange(frameHsv, lowLimit, highLimit)
    backprojectedFrame &= mask

    ret, window = cv.CamShift(backprojectedFrame, windowC, terminationCriteria)


    points = cv.boxPoints(ret)
    points = np.int0(points)
    frame = cv.polylines(img2, [points], True, 255, 2)
    #showMe(frame,0)
    return frame


