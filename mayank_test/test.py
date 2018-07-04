# import cv2                              
# import numpy as np                           #importing libraries
# cap = cv2.VideoCapture(0)                #creating camera object
# while( cap.isOpened() ) :
#    ret,img = cap.read()   
#    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#    blur = cv2.GaussianBlur(gray,(5,5),0)
#    ret,thresh1 = cv2.threshold(blur,70,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)                      #reading the frames
   
#    _, contours, hierarchy = cv2.findContours(thresh1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
  
#    max_area=0
#    for i in range(len(contours)):
#       cnt=contours[i]
#       area = cv2.contourArea(cnt)
#       if(area>max_area):
#           max_area=area
#           ci=i
#    cnt=contours[ci]
#    hull = cv2.convexHull(cnt)
#    drawing = np.zeros(img.shape,np.uint8)
#    cv2.drawContours(drawing,[cnt],0,(0,255,0),2)
#    cv2.drawContours(drawing,[hull],0,(0,0,255),2)

#    hull = cv2.convexHull(cnt,returnPoints = False)
#    defects = cv2.convexityDefects(cnt,hull)

#    mind=0
#    maxd=0
#    centr=0
#    for i in range(defects.shape[0]):
#       s,e,f,d = defects[i,0]
#       start = tuple(cnt[s][0])
#       end = tuple(cnt[e][0])
#       far = tuple(cnt[f][0])
#       dist = cv2.pointPolygonTest(cnt,centr,True)
#       cv2.line(img,start,end,[0,255,0],2)                
#       cv2.circle(img,far,5,[0,0,255],-1)
#       print(i)

#    cv2.imshow('input',drawing)                  #displaying the frames
#    k = cv2.waitKey(10)
#    if k == 27:
#    	break
import cv2
import numpy as np
cap = cv2.VideoCapture(0)
is_background_captured=False
bgSubThreshold = 50
learningRate = 0


def removeBG(frame):
    fgmask = bgModel.apply(frame,learningRate=learningRate)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # res = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    return res


while( cap.isOpened() ) :
    if is_background_captured:
        ret,img = cap.read()
        img = removeBG(img)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray,(5,5),0)
        ret,thresh1 = cv2.threshold(blur,70,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        cv2.imshow('output',thresh1)


        _, contours, hierarchy = cv2.findContours(thresh1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        drawing = np.zeros(img.shape,np.uint8)

        max_area=0
       
        for i in range(len(contours)):
                cnt=contours[i]
                area = cv2.contourArea(cnt)
                if(area>max_area):
                    max_area=area
                    ci=i
        cnt=contours[ci]
        hull = cv2.convexHull(cnt)
        moments = cv2.moments(cnt)
        if moments['m00']!=0:
                    cx = int(moments['m10']/moments['m00']) # cx = M10/M00
                    cy = int(moments['m01']/moments['m00']) # cy = M01/M00
                  
        centr=(cx,cy)       
        cv2.circle(img,centr,5,[0,0,255],2)       
        cv2.drawContours(drawing,[cnt],0,(0,255,0),2) 
        cv2.drawContours(drawing,[hull],0,(0,0,255),2) 
              
        cnt = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
        hull = cv2.convexHull(cnt,returnPoints = False)
        
        if(1):
                   defects = cv2.convexityDefects(cnt,hull)
                   mind=0
                   maxd=0
                   if defects is not None:
                       for i in range(defects.shape[0]):
                            s,e,f,d = defects[i,0]
                            start = tuple(cnt[s][0])
                            end = tuple(cnt[e][0])
                            far = tuple(cnt[f][0])
                            dist = cv2.pointPolygonTest(cnt,centr,True)
                            cv2.line(img,start,end,[0,255,0],2)
                            
                            cv2.circle(img,far,5,[0,0,255],-1)
                       print(i)
        cv2.imshow('output',drawing)
        cv2.imshow('input',img)
    else:
        bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)
        is_background_captured = True
        print( '!!!Background Captured!!!')
                    
    k = cv2.waitKey(10)
    if k == 27:
        break
