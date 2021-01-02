import cv2
import numpy as np
import math


class HandProcess():
    def __init__(self):
        pass

    def thresholdWithYCC(self,img):
        # convert to hsv value for extract skin tone
        ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
        # lower = np.array([0, 133, 77])
        # upper = np.array([255, 173, 127])
        # lower = np.array([0, 120, 60])
        # upper = np.array([255, 180, 130])
        lower = np.array([0, 133, 77])
        upper = np.array([255, 200, 150])
        # lower = np.array([0, 120, 77])
        # upper = np.array([255, 200, 150])
        skintone = cv2.inRange(ycc, lower, upper)

        skintone_mask = cv2.bitwise_and(img, img, mask=skintone)  # extract skin tone
        # cv2.imshow("skintone_mask",skintone_mask)

        # convert to gray scale for extract background and Light luminate
        gray = cv2.cvtColor(skintone_mask, cv2.COLOR_BGR2GRAY)      #convert to gray scale
        valueofGaussian = (13, 13)
        gray = cv2.GaussianBlur(gray, valueofGaussian, 0)           # used GaussianBlur for smote edge
        # cv2.imshow("gray_mask", gray)
        res, thres = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY )     #Extract hand from backgroud + cv2.THRESH_OTSU
        # cv2.imshow("GraySkintone", thres)

        mask_skinTone_binary = self.fillHold(thres)
        real_skinTone = cv2.bitwise_and(img, img, mask=mask_skinTone_binary)
        # cv2.imshow("real_skinTone", real_skinTone)
        return mask_skinTone_binary,real_skinTone

    def fillHold(self,img):
        im_floodfill = img.copy()
        h,w = img.shape[:2]
        mask = np.zeros((h+2,w+2),np.uint8)
        cv2.floodFill(im_floodfill,mask,(0,0),255)      # Fill with white screen
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)    # invert black to white for mask
        im_out = img|im_floodfill_inv

        # cv2.imshow(" Image", img)
        # cv2.imshow("im_floodfill Image", im_floodfill)
        # cv2.imshow("Inverted Floodfilled Image", im_floodfill_inv)
        # cv2.imshow("Foreground", im_out)

        return im_out

    def createImageToRealSize(self,ROI,size=(200,200,3)):
        height, width = ROI.shape[:2]
        max =np.amax([height,width])
        if(max>size[0]):
            ratio = float(size[0])/max
            # print(ratio)
            ROI = cv2.resize(ROI,(0,0),fx=ratio,fy=ratio)
            height, width = ROI.shape[:2]

        start_h = int((size[0]-height)/2)
        start_w = int((size[1]-width)/2)
        NewImg = np.ones(size,dtype=np.uint8)
        # cv2.imshow("ROI", ROI)
        NewImg[start_h:start_h+int(height), start_w:start_w+int(width)] = ROI
        # cv2.imshow("gray", NewImg)
        return NewImg

    def process(self,img,size=200):
        thresh_skin,skincolor = self.thresholdWithYCC(img)
        #cv2.imshow("skinColor",thresh_skin)
        #cv2.waitKey(0)
        contours , hierarchy = cv2.findContours(thresh_skin,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        areas = [cv2.contourArea(Contour) for Contour in contours]          #find The area in controurArea list
        # print(areas)
        if(areas != []):
            max_index = np.argmax(areas)                                        #find The largest in controurArea list
            newcontour = np.copy(contours)

            cnt = contours[max_index]                                           # get theposition of controur Area
            blank_image = np.zeros((img.shape[:2]), np.uint8)


            cv2.drawContours(blank_image, contours,max_index, (255,255, 255),-1)
            ret,th = cv2.threshold(blank_image, 127, 255, cv2.THRESH_BINARY)
            # cv2.imshow("MarkLayer", th)
            skintone_mask = cv2.bitwise_and(img, img, mask=th)  # extract skin tone

            # cv2.imshow("MarkImage", skintone_mask)
            # cv2.waitKey(0)
            x, y, w, h = cv2.boundingRect(cnt)
            # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)    #(x1,y1),(x2,y2)
            ROI = skincolor[y:y + h, x:x + w]  #[y1:y2, x1:x2]
            # cv2.imshow("ROI",ROI)
            ROI = self.createImageToRealSize(ROI,size=(size,size,3))
            
            M = cv2.moments(th)
             # calculate x,y coordinate of center
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            x, y, w, h = 0,0,0,0
            ROI = self.createImageToRealSize(img, size=(size, size, 3))
            cX,cY = 0,0
            th = np.zeros((200,200,3),dtype=np.uint8)
        return ROI,(x, y, w, h),(cX,cY),th

if __name__ == "__main__":

    image2 = cv2.imread("P1130026.jpg")
    # cv2.imshow("Real Image",image2)
    handPro = HandProcess()
    ROI = handPro.process(image2,64)
    cv2.imshow("ROI", ROI)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
 