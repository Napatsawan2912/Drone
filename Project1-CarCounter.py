from ultralytics import YOLO
import cv2
import cvzone
import math
from tracker import *

#cap = cv2.VideoCapture(0) #for webcam
#cap.set(3,640)
#cap.set(4,480)
cap = cv2.VideoCapture("videos/test.mp4") #for video

model = YOLO('Yolov8-Weights/best.pt') #?
classNames = ['Apple', 'Banana', 'Kiwi', 'Orange', 'Pear']

# Initialize the MultiTracker
tracker = Tracker() #tracker.py
limits =[450, 0, 450, 800] #tracker.py
totalCount = [] #tracker.py

# Read the first frame to get dimensions
#success, img = cap.read()
#if not success:
    #print("Failed to read video")
    #cap.release()
    #cv2.destroyAllWindows()
    #exit(0)

# Load and resize mask
#mask = cv2.imread('pictures/mask-950x480.png')
#mask = cv2.resize(mask, (img.shape[1], img.shape[0]))  # Resize mask to match video frame

while True:
    success, img = cap.read()
    #imgRegion = cv2.bitwise_and(img)
    #imgGraphics = cv2.imread('pictures/graphics.png', cv2.IMREAD_UNCHANGED)
    #img = cvzone.overlayPNG(img)
    #results = model(img, stream=True)
    list = [] #tracker.py
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding box
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
            w,h = x2-x1, y2-y1
            
            #Connfidence
            conf = math.ceil((box.conf[0]*100))/100

            # Class names
            cls = int(box.cls[0])
            # print(conf)
            currentClass = classNames[cls]
            if (currentClass == 'Orange') and conf>0.8:
                list.append([x1,y1,x2,y2]) #tracker.py
                cvzone.cornerRect(img,(x1,y1,w,h),l=9)
                cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0,x1), max(35,y1)), scale=1.0, thickness=1, offset=3)

    # Convert BGR image to RGB
    # img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)        
    # cv2.imshow("Image",imgRegion)
    boxes_id = tracker.update(list) #tracker.py
    cv2.line(img, (limits[0],limits[1]),(limits[2],limits[3]),(0,0,255),5) #tracker.py
    for i, newbox in enumerate(boxes_id): #tracker.py
        # x, y, w, h = [int(v) for v in newbox]
        # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2, 1)
        # cv2.putText(img, f"ID {i}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        x3,y3,x4,y4,id = newbox     
        cv2.rectangle(img, (x3, y3), (x4, y4), (255, 0, 255), 2, 1)
        cv2.putText(img, f"ID {i}", (x3, y3 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cx,cy = x3+abs(x3-x4)//2, y3+abs(y3-y4)//2
        cv2.circle(img,(cx,cy),5,(255,0,255),cv2.FILLED)
        
        if limits[0]-15 < cx < limits[2]+15 and limits[1] < cy < limits[3]:
            if totalCount.count(id) == 0:
                totalCount.append(id)
                cv2.line(img, (limits[0],limits[1]),(limits[2],limits[3]),(0,255,0),5)
    
    # cvzone.putTextRect(img, f'Count:{len(totalCount)}', (50,50))
    cv2.putText(img,str(len(totalCount)),(255,100),cv2.FONT_HERSHEY_PLAIN,5,(50,50,255),8) #tracker.py
    #cv2.putText(img,str(totalCount),(255,100),cv2.FONT_HERSHEY_PLAIN,5,(50,50,255),8)
    cv2.imshow("Image Region",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting loop...")
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
