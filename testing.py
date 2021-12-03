#testing obj detection with similar color
import cv2
import numpy as np
import pafy #used for running youtube video 

#webcame = cv2.VideoCapture(0)
capture = cv2.VideoCapture('kart_video.mp4')


while True:
    #choose capture or webcame
    #_, img = webcame.read()
    _, img = capture.read()

    #choose image to read
    #img = cv2.imread('road.jpg')
    #img = cv2.imread('fullTrack.jpg')
    #img = cv2.imread('raceTrack.jpg')

    #convert to hsv colorspace
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    #lower bound and upper bound for color grey
    grey_lower = np.array([0, 5, 50])
    grey_upper = np.array([179, 50, 225])
    #boudns for green but not used
    green_lower = np.array([50, 20, 20])
    green_upper = np.array([100, 225, 225])

    #creating mask threshold
    mask_grey = cv2.inRange(hsv_img, grey_lower, grey_upper)

    #removing noise
    kernel = np.ones((7, 7), np.uint8) #define kernel size

    mask_grey = cv2.morphologyEx(mask_grey, cv2.MORPH_CLOSE, kernel) #remove black noise from white region
    mask_grey = cv2.morphologyEx(mask_grey, cv2.MORPH_OPEN, kernel) #remove white noise from black region
    
    #combined img
    combine_img = cv2.bitwise_and(img, img, mask= mask_grey)

    #finding contours for mask img (similar to finding edges)
    contours, hierarchy = cv2.findContours(mask_grey.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #draws contour on combined img
    contourOutput = cv2.drawContours(combine_img, contours, -1, (0, 0, 255), 3)
    
    #blur img for better resutls
    img_blur = cv2.GaussianBlur(mask_grey, (3,3), 0)

    #Canny edge detection
    edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)

    #Combining canny edge with reg img
    redImg = np.zeros(img.shape, img.dtype)
    redImg[:,:] = (0, 0, 255)
    redMask = cv2.bitwise_and(redImg, redImg, mask=edges)
    overlayImg = cv2.addWeighted(redMask, 1, img, 1, 0)

    #Choose which output to show
    #cv2.imshow("Regular Image", img)
    #cv2.imshow("mask_image", mask_grey)
    #cv2.imshow("Contour Lingings", contourOutput)
    #cv2.imshow("Canny Edge Detection Lingings", edges)
    cv2.imshow("Blended", overlayImg)

    #press shift+Q to exit
    if cv2.waitKey(1) & 0xFF == ord('Q'):
        break

#webcame.release()
capture.release()


