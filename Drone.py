import cv2
import numpy as np
import math
from cvzone.HandTrackingModule import HandDetector
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from win32api import GetKeyState
from djitellopy import tello
import time

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Initialize the HandDetector
detector = HandDetector(detectionCon=0.8, maxHands=2)
offset = 20
imgSize = 300
# folder = "dataset/train/A"
# counter = 0
# print("xxxxxx")

# --- Load the Keras Model ---
# model = load_model('model2/abcd_model2.keras')  # Replace with your model file
# labels = ["A", "B", "C", "D"]  # Replace with your class labels
model1 = load_model('models/hand_gesture_control_8a.keras')  # Replace with your model file
# labels = ["Ascending", "Descending", "Pitch_Backward", "Pitch_forward","Roll_Left","Roll_Right"]  # Replace with your class labels
labels = ["Ascending", "Descending", "Pitch_Backward", "Pitch_Forward","Roll_Left","Roll_Right","Yaw_Left","Yaw_Right"]  # Replace with your class labels

# --- Get the model's input shape ---
input_shape = model1.input_shape[1:]  
predicted_class = 0
gesture_control = 0

me = tello.Tello()
me.connect()
print(me.get_battery())
me.streamon()
time.sleep(1)

def getGastureInput(ind):
    lr, fb, ud, yv = 0, 0, 0, 0
    speed = 30
    if ind=="Roll_Left": # "ROLL_LEFT": #kp.getKey("LEFT"):
        lr = -speed
    elif ind=="Roll_Right": #"ROLL_RIGHT":
        lr = speed
    if ind=="Pitch_Forward": #"PITCH_FORWARD":
        fb = speed
    elif ind=="Pitch_Backward": #"PITCH_BACKWARD":
        fb = -speed
    if ind=="Ascending": #"ASCENDING":#kp.getKey("w"):
        ud = speed
    elif ind=="Descending": #"DESCENDING":#kp.getKey("s"):
        ud = -speed
    if ind=="Yaw_Left": #"YAW_LEFT": #kp.getKey("a"):
        yv = -speed
    elif ind=="Yaw_Right": #"YAW_RIGHT": #kp.getKey("d"):
        yv = speed
    # if ind=="HOVER": #kp.getKey("e"):
    #     me.takeoff()
        # print("Drone-Hover")
    if ind=="CAPTURE": #kp.getKey("z"):
        # cv2.imwrite(f'Resources/Images/{time.time()}.jpg', img)
        print("Capture")
        time.sleep(0.5)
   
    return [lr, fb, ud, yv]

def key_down(key):
    state = GetKeyState(key)
    if (state != 0) and (state != 1):
        return True
    else:
        return False


def getKeyboardInput():
    lr, fb, ud, yv = 0, 0, 0, 0
    speed = 50
    
    if key_down(37): # Arrow Left
        lr = -speed
        # print("Left")
    if key_down(0x26): # Arrow Up
        fb = speed
    if key_down(0x27): # Arrow Right
        lr = speed
    if key_down(0x28): # Arrow Down
        fb = -speed

    if key_down(int(ord("w"))-0x20): 
        ud = speed
        # print("Ascending")
    elif key_down(int(ord("s"))-0x20):
        ud = -speed
        # print("Descending")
    if key_down(int(ord("a"))-0x20):
        yv = -speed
        # print("Yaw Left")
    elif key_down(int(ord("d"))-0x20):
        yv = speed
        # print("Yaw Right")
    if key_down(int(ord("l"))-0x20):
        me.land()
        time.sleep(3)
        print("Landing")
    if key_down(int(ord("e"))-0x20):
        me.takeoff()
        # print("Takeoff")
    if key_down(int(ord("z"))-0x20):
        cv2.imwrite(f'pictures/{time.time()}.jpg', img)
        time.sleep(0.3)
        # print("Capture")
    return [lr, fb, ud, yv]


while True:
    # Capture frame from webcam
    success, img = cap.read()
    img = cv2.flip(img,1)
    imgOutput = img.copy()

    gesture_control = 0
    
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
            gesture_control = 1
            aspectRatio = h/w
            if aspectRatio > 1:
                k = imgSize/h
                wCal= math.ceil(k*w)
                imgResize = cv2.resize(imgCrop,(wCal,imgSize))
                wGap = math.ceil((imgSize-wCal)/2)

                imgWhite[0:imgResize.shape[0],wGap:wCal+wGap] = imgResize

                # --- Preprocess for Keras ---
                imgCrop = cv2.cvtColor(imgWhite, cv2.COLOR_BGR2RGB)  # Convert to RGB
                imgCrop = image.img_to_array(imgCrop)
                imgCrop = imgCrop / 255.0  # Normalize
                imgCrop = tf.expand_dims(imgCrop, axis=0)  # Add batch dimension

                # --- Resize to model's input shape (Corrected) ---
                imgCrop = tf.image.resize(imgCrop, (input_shape[0], input_shape[1])) 
                # --- Resize to (300, 300) ---
                imgCrop = tf.image.resize(imgCrop, (300, 300)) 

                # --- Make Prediction ---

                prediction = model1.predict(imgCrop)
                predicted_class = tf.math.argmax(prediction[0]).numpy()
                # print(f"Predicted Class: {labels[predicted_class]}")
                vals = getGastureInput(labels[predicted_class])
                me.send_rc_control(vals[0], vals[1], vals[2], vals[3])
                # print(vals)

            else:
                k = imgSize/w
                hCal= math.ceil(k*h)
                imgResize = cv2.resize(imgCrop,(imgSize,hCal))
                hGap = math.ceil((imgSize-hCal)/2)

                imgWhite[hGap:hCal+hGap, : ] = imgResize
 
                # --- Preprocess for Keras ---
                imgCrop = cv2.cvtColor(imgWhite, cv2.COLOR_BGR2RGB)  # Convert to RGB
                imgCrop = image.img_to_array(imgCrop)
                imgCrop = imgCrop / 255.0  # Normalize
                imgCrop = tf.expand_dims(imgCrop, axis=0)  # Add batch dimension

                # --- Resize to model's input shape (Corrected) ---
                imgCrop = tf.image.resize(imgCrop, (input_shape[0], input_shape[1])) 
                # --- Resize to (300, 300) ---
                imgCrop = tf.image.resize(imgCrop, (300, 300)) 

                # --- Make Prediction ---

                prediction = model1.predict(imgCrop)
                predicted_class = tf.math.argmax(prediction[0]).numpy()
                # print(f"Predicted Class: {labels[predicted_class]}")
                vals = getGastureInput(labels[predicted_class])
                me.send_rc_control(vals[0], vals[1], vals[2], vals[3])
                # print(vals)
                         
            cv2.rectangle(imgOutput, (x, y), (x + w, y + h), (255, 0, 255), 4)        
            cv2.putText(imgOutput, labels[predicted_class], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.3, (255, 100,100), 2)

            # cv2.imshow("ImageCrop",imgCrop)
            cv2.imshow("ImageWhite",imgWhite)
            # if cv2.waitKey(1) == ord("s"):
            #     counter +=1
            #     # fname = 'img'+str(counter)
            #     cv2.imwrite(f'{folder}/Image_{time.time()}.jpg',imgWhite)
            #     # cv2.imwrite(f'Image_{counter}.jpg',imgWhite)
            #     print(counter)
            #     time.sleep(0.1)
        except Exception as e:
            gesture_control = 0
            # Code to handle other exceptions
            print(f"An unexpected error occurred: {e}")
            continue

    if gesture_control == 0:
        vals = getKeyboardInput() 
        me.send_rc_control(vals[0], vals[1], vals[2], vals[3])
    # Display the image
    cv2.imshow("Control Image", imgOutput)

    capture_img = me.get_frame_read().frame
    imgOutput = cv2.resize(capture_img, (360, 240))
    imgOutput = cv2.cvtColor(imgOutput, cv2.COLOR_BGR2RGB)
    cv2.imshow("Drone Image:", imgOutput)


    # Exit the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('t'):
        break
# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
