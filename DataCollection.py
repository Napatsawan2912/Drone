import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Initialize the HandDetector
detector = HandDetector(detectionCon=0.8, maxHands=2)
offset = 20
imgSize = 300
folder = "dataset/raw_data/Ascending"
counter = 0
print("xxxxxx")
while True:
    # Capture frame from webcam
    success, img = cap.read()
    img = cv2.flip(img,1)
   
    
    if not success:
        break

    # Detect hands in the image
    hands, img = detector.findHands(img)  # With draw
    
    # hands = detector.findHands(img, draw=False) # Without draw

    # Extract hand landmarks if any hand is detected
    if hands:
        # Get the first hand detected
        hand1 = hands[0]
        # Get Hand landmarks
        lmList = hand1['lmList']  # List of 21 Landmark points
        # Get bounding box info
        bbox = hand1['bbox']  # Bounding box
        x,y,w,h = hand1['bbox']
        # Get the center of the hand
        centerPoint = hand1['center']  # Center of hand
        # Get the type of hand (left or right)
        handType = hand1['type']  # Hand type

        # Print landmarks or other information if needed
        # print(f'Hand Type: {handType}')
        # print(f'Hand Landmarks: {lmList}')
        # print(f'Bounding Box: {bbox}')
        # print(f'Center Point: {centerPoint}')
        # print(f'X:{x} Y:{y} W:{w} H:{h}')

        # You can also process each landmark to do other tasks, e.g., detect gestures

        x,y,w,h = hand1['bbox']

        imgWhite = np.ones((imgSize,imgSize,3),np.uint8)*255
        imgCrop = img[y-offset:y+h+offset,x-offset:x+w+offset]

        try:
            aspectRatio = h/w
            if aspectRatio > 1:
                k = imgSize/h
                wCal= math.ceil(k*w)
                imgResize = cv2.resize(imgCrop,(wCal,imgSize))
                wGap = math.ceil((imgSize-wCal)/2)

                imgWhite[0:imgResize.shape[0],wGap:wCal+wGap] = imgResize

            else:
                k = imgSize/w
                hCal= math.ceil(k*h)
                imgResize = cv2.resize(imgCrop,(imgSize,hCal))
                hGap = math.ceil((imgSize-hCal)/2)

                imgWhite[hGap:hCal+hGap, : ] = imgResize
            

            cv2.imshow("ImageCrop",imgCrop)
            cv2.imshow("ImageWhite",imgWhite)
            if cv2.waitKey(1) == ord("s"):
                counter +=1
                # fname = 'img'+str(counter)
                cv2.imwrite(f'{folder}/Image_{time.time()}.jpg',imgWhite)
                # cv2.imwrite(f'Image_{counter}.jpg',imgWhite)
                print(counter)
                time.sleep(0.1)
        except Exception as e:
            # Code to handle other exceptions
            print(f"An unexpected error occurred: {e}")
            continue

    # Display the image
    # img = cv2.flip(img,1)
    cv2.imshow("Image", img)

    # Exit the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
