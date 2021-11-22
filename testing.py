#testing obj detection with similar color
import cv2
import numpy as np
import pafy #used for running youtube video but isnt working 

#create Video
url = 'https://www.youtube.com/watch?v=Gbod4z8LHQ4'
video = pafy.new(url)
best_vid = video.getbest() #should be adding a parameter preftype='webm'

#trying to see if there is the webm extention bc needed for 
streams = video.streams
for s in streams:
    print(s.resolution, s.extension)
    print("")

#webcame = cv2.VideoCapture(0)
#capture = cv2.VideoCapture(best_vid.url)


while True:
    #_, img = webcame.read()
    #_, img = capture.read()

    #reading an image
    img = cv2.imread('road.jpg')
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
    output = cv2.drawContours(combine_img, contours, -1, (0, 0, 255), 3)
    
    #blur img for better resutls
    img_blur = cv2.GaussianBlur(mask_grey, (3,3), 0) #using Canny on grey mask, idk if combined better?

    #Canny edge detection
    edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)

    #showing the output
    cv2.imshow("Regular Image", img)
    #cv2.imshow("mask_image", mask_grey)
    cv2.imshow("Contour Lingings", output)
    cv2.imshow("Canny Edge Detection Lingings", edges)



    if cv2.waitKey(1) & 0xFF == ord('Q'): #image window will be opened until any key is pressed
        break

#webcame.release()
#capture.release()


