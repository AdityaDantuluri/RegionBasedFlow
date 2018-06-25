import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle
def toGray(imageIn):
    return cv.cvtColor(imageIn, cv.COLOR_RGB2GRAY);

def G2R(imageIn):
    return cv.cvtColor(imageIn, cv.COLOR_GRAY2RGB);

def showMe(imageIn, n):
    cv.destroyAllWindows()
    print imageIn.shape
    cv.imshow('',imageIn)
    k = cv.waitKey(n)

def showMany(images):
    count = 0;
    for img in images:
        cv.imshow(str(count), img)
        count += 1
    cv.waitKey(0)
    cv.destroyAllWindows();

def OpenClose(imageIn,r):
    I = imageIn
    kernel= cv.getStructuringElement(cv.MORPH_ELLIPSE,(r,r))

    I = cv.morphologyEx(I, cv.MORPH_OPEN, kernel)
    I = cv.morphologyEx(I, cv.MORPH_CLOSE, kernel)
    return I

def closer(imageIn,r):
    I = imageIn
    kernel= cv.getStructuringElement(cv.MORPH_ELLIPSE,(r,r))
    I = cv.morphologyEx(I, cv.MORPH_CLOSE, kernel)
    return I

def opener(imageIn,r):
    I = imageIn
    kernel= cv.getStructuringElement(cv.MORPH_ELLIPSE,(r,r))
    I = cv.morphologyEx(I, cv.MORPH_OPEN, kernel)
    return I

def blur(imageIn,r):
    return cv.GaussianBlur(imageIn, (r,r),0)

def multMed(imageIn,n,r):
    I = imageIn
    for i in range(n):
        I = cv.medianBlur(I,r)
    return I;

def BWEdged(InI,t):
    gray =cv.cvtColor(InI, cv.COLOR_RGB2GRAY);
    edges = cv.Canny(gray,t,t)
    edges = G2R(edges)
    return edges;

def CEdged(InI,t):
    b,g,r = cv.split(InI)
    bgr = [b,g,r]
    OutI = bgr;
    for i in range(0,3):
        I = bgr[i]
        edges = cv.Canny(I,t,t)
        OutI[i] = edges;
    img = OutI[0]|OutI[1]|OutI[2]
    #gray =cv.cvtColor(img, cv.COLOR_RGB2GRAY);
    #_, img = cv.threshold(gray,10,255,cv.THRESH_BINARY)
    th = G2R(img)
    return th;

def HEdged(InI,t):
    hsvI = C2H(InI)
    h,s,v = cv.split(hsvI)
    hsv = [h,s,v]
    OutI = hsv;
    for i in range(0,3):
        I = hsv[i]
        edges = cv.Canny(I,t,t)
        OutI[i] = edges;
    img = OutI[0]|OutI[1]|OutI[2]
    color = G2R(img)
    #gray =cv.cvtColor(color, cv.COLOR_RGB2GRAY);
    #_, img = cv.threshold(gray,10,255,cv.THRESH_BINARY)
    th = G2R(img)
    return th;

def MeanS(img, n):
    return cv.pyrMeanShiftFiltering(img,n,n)

def C2H(img):
    return cv.cvtColor(img, cv.COLOR_BGR2HSV)

def H2C(img):
    return cv.cvtColor(img, cv.COLOR_HSV2BGR)

def HSVRep(hsvI, hsv1, hsv2, hsv3):
    hsvO = hsvI
    hsvO[:,:,0] = hsv1
    hsvO[:,:,1] = hsv2
    hsvO[:,:,2] = hsv3
    return hsvO

def Skeleton(img,r):
    img = toGray(img)
    size = np.size(img)
    skel = np.zeros(img.shape,np.uint8)
 
    ret,img = cv.threshold(img,127,255,0)
    element = cv.getStructuringElement(cv.MORPH_CROSS,(r,r))
    done = False
 
    while( not done):
        eroded = cv.erode(img,element)
        temp = cv.dilate(eroded,element)
        temp = cv.subtract(img,temp)
        skel = cv.bitwise_or(skel,temp)
        img = eroded.copy()
        zeros = size - cv.countNonZero(img)
        if zeros==size:
            done = True

    return G2R(skel)

def dilator(img, n):
    element = cv.getStructuringElement(cv.MORPH_RECT,(n,n))
    return cv.dilate(img,element)

def closePix(img):
    height, width,_ = img.shape
    mask = np.zeros(img.shape, np.uint8)
    for x in range(1,height-1):
        for y in range(1,width-1):
            up = img[x, y+1,1]
            down = img[x, y-1,1]
            right = img[x+1, y,1]
            left = img[x-1, y,1]
            if( (up != 0) & (down != 0)):
                mask[x,y,:] = 255
            if( (left != 0) & (right != 0)):
                mask[x,y,:] = 255
    img = img | mask
    return img
def closePix2(img):
    element = cv.getStructuringElement(cv.MORPH_CROSS,(3,3))
    helement = np.array([[0,0,0], [1,1,1], [0,0,0]], np.uint8)
    velement = np.rot90(helement,1)
    dilated = cv.dilate(img,element)
    eroded1 = cv.erode(dilated,helement)
    eroded2 = cv.erode(dilated,velement)
    eroded = eroded1 | eroded2
    return eroded

def flooder(img, (x,y), r):
    #img = toGray(img)
    th, im_th = cv.threshold(img, 125, 255, cv.THRESH_BINARY);
  
    im_floodfill = im_th.copy()

   
    h, w = im_th.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    
    cv.floodFill(im_floodfill, mask, (x,y), r);

    
    im_floodfill_inv = cv.bitwise_not(im_floodfill)

    
    img = im_th | im_floodfill_inv
    img = cv.bitwise_not(img)
    #img = G2R(img)
    return img;

def edge2Segmented(img):
    img = toGray(img)
    segments = []
    mask = np.zeros(img.shape)
    label = 1;
    height, width = img.shape
    for i in range(height):
        for j in range(width):
            if img.item(i,j)== 0:
                flood = flooder(img,(j,i),1)
                img+= flood*label
                floodseg = flood
                segments.append(floodseg)
                #showMe(floodseg,1)
                label+=1
                pass
    return img,segments

def blend_transparent(face_img, overlay_t_img):
    return cv.addWeighted(face_img, 0.9, overlay_t_img, 0.5, 0)

def liner(img):
    d = 10;
    height, width,_ = img.shape
    hInc = int(height/d)
    wInc = int(width/d)
    for i in range(d):
        img = cv.line(img,(0,i*hInc), (width, i*hInc), (255,255,255),1)
        img = cv.line(img,(i*wInc, 0), (i*wInc, height), (255,255,255),1)
    return img

def SegImg(Segs):
    shuffle(Segs)
    num = len(Segs)
    print (num)
    inc = int(255/num)+1
    print (inc)
    width,height = Segs[0].shape
    img= np.zeros(Segs[0].shape, np.uint8)
    lab = 0
    for i in range(num):
        img += np.uint8((Segs[i]*lab))
        lab +=1 
    return img

def GetRect(img):
    #gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(img, 0, 255, cv.THRESH_BINARY)
    im2, contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    rect = cv.boundingRect(contours[0])
    x,y,w,h = rect
    img = cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    return img, rect

def GetRRect(img):
    #gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(img, 0, 255, cv.THRESH_BINARY)
    im2, contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    rect = cv.minAreaRect(contours[0])
    box = cv.boxPoints(rect)
    box = np.int0(box)
    img = cv.drawContours(img,[box],0,(0,0,255),2)
    return img, rect

def getROI(image, rect):
    src = image
    print rect
    x = rect[0]
    y = rect[1]
    w = rect[2]
    h = rect[3]
    #angle = rect[2]
    #cyc = int(rect[0][0])
    #cxc = int(rect[0][1])
    #ch = int(rect[1][0]/2)
    #cw = int(rect[1][1]/2)
    #cx = (cxc-cw)
    #cw = cw*2
    #cy = (cyc-ch)
    #ch = ch*2
    #center = cyc,cxc
    #img2 = src[cx:(cx+cw), cy:(cy+ch)]
    #M = cv.getRotationMatrix2D(center, angle, 1.0)
    #width, height,_ = src.shape
    #rotated = cv.warpAffine(src, M, (height, width))
    #cropped = rotated[cx:(cx+cw), cy:(cy+ch)]
    x2 = (x+w)
    y2 = (y+h)
    cropped = src[y:y2, x:x2]
    return cropped

def rotateImage(image, angle):
      image_center = tuple(np.array(image.shape[1::-1]) / 2)
      rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
      result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
      return result

def overlayRect(img, rect):
    x,y,w,h = rect
    return cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

def overlayRRect(img, rect):
    box = cv.boxPoints(rect)
    box = np.int0(box)
    return cv.drawContours(img,[box],0,(0,0,255),2)