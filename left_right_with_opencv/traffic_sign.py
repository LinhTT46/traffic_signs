import cv2
import numpy as np


class Traffic_Sign:
    def __init__(self):
        # label of traffic sign detected
        self.sign = 0
        self.min_area = 300

        # range of traffic_sign background
        self.lowerBound = np.array([98, 109, 20])
        self.upperBound = np.array([112, 255, 255])

        self.area_left = 0
        self.area_right = 0
        self.area_left_before = 0
        self.area_right_before = 0

    # find contours of traffic sign
    def find_contour(self, image):
        kernelOpen = np.ones((5, 5))
        kernelClose = np.ones((20, 20))

        # convert BGR to HSV
        imgHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # create the Mask
        mask = cv2.inRange(imgHSV, self.lowerBound, self.upperBound)

        # morphology to remove noise
        maskOpen = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernelOpen)
        maskClose = cv2.morphologyEx(maskOpen, cv2.MORPH_CLOSE, kernelClose)

        maskFinal = maskClose
        _, conts, _ = cv2.findContours(maskFinal.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        return conts

    def get_traffic_sign(self, image):
        txt = '' # labels of traffic signs
        image = image.copy()

        height, width = image.shape[:2]
        SKYLINE = int(height * 0.45)
        
        # the upper region of image
        roi = image[0:SKYLINE, 0:width]
        
        # contours of the roi
        conts = self.find_contour(roi)

        if len(conts) == 0:  # no_sign
            # self.area of the rois contain two upper half of traffic sign
            self.area_left = 0
            self.area_right = 0
            self.area_left_before = 0
            self.area_right_before = 0
            self.sign = 0

        for i in range(len(conts)):
            # rectangle contains traffic sign
            x, y, w, h = cv2.boundingRect(conts[i])
            
            # (cX,cY) is a center point of 1/4 bounding box.
            cX = int(w/2)+x
            cY = int(h/4)+y

            # remove small independent objects
            if (cv2.contourArea(conts[i]) > self.min_area):
                # draw the bounding box 
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
                
                # the upper left conner of rectangle
                imgl = image[y:cY, x:cX]
                conts1 = self.find_contour(imgl)

                # the upper right conner of rectangle
                imgr = image[y:cY, cX:x+w]
                conts2 = self.find_contour(imgr)

                if len(conts1) != 0 and len(conts2) != 0:
                    # sum self.area each contour of each part
                    self.area_left = self.area_left + \
                        cv2.contourArea(conts1[i])
                    self.area_right = self.area_right + \
                        cv2.contourArea(conts2[i])
                    
                    if self.area_left > self.area_right and self.area_left > self.area_left_before:  # right
                        self.sign = 1
                        self.area_left_before = self.area_left
                        txt = 'right'

                    elif self.area_left < self.area_right and self.area_right > self.area_right_before:  # left
                        self.sign = -1
                        self.area_right_before = self.area_right
                        txt = 'left'

                    cv2.putText(image,txt,(x,y-1),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,255),2,cv2.LINE_AA)
                    
        return self.sign, image

if __name__ == "__main__":

    cap =  cv2.VideoCapture('unity_v2_sample.avi')
    trafficsign = Traffic_Sign()

    while(cap.isOpened):
        _, img = cap.read()
        
        sign, image = trafficsign.get_traffic_sign(img)
        cv2.imshow('img', image)
        
        key=cv2.waitKey(9)
        if (key==27):
            cv2.destroyAllWindows()