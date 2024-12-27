from win32api import GetKeyState
from djitellopy import tello
import time
import cv2

me = tello.Tello()
me.connect()
print(me.get_battery())
me.streamon()
time.sleep(1)

# cap = cv2.VideoCapture(0)

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
        print("Left")
    if key_down(0x26): # Arrow Up
        fb = speed
        print("Up")
    if key_down(0x27): # Arrow Right
        lr = speed
        print("Right")
    if key_down(0x28): # Arrow Down
        fb = -speed
        print("Down")
    if key_down(int(ord("w"))-0x20): 
        ud = speed
        print("Ascending")
    elif key_down(int(ord("s"))-0x20):
        ud = -speed
        print("Descending")
    if key_down(int(ord("a"))-0x20):
        yv = -speed
        print("Yaw Left")
    elif key_down(int(ord("d"))-0x20):
        yv = speed
        print("Yaw Right")
    if key_down(int(ord("l"))-0x20):
        me.land()
        # time.sleep(3)
        print("Landing")
    if key_down(int(ord("e"))-0x20):
        me.takeoff()
        print("Takeoff")
    if key_down(int(ord("z"))-0x20):
        cv2.imwrite(f'pictures/{time.time()}.jpg', img) #?
        time.sleep(0.3)
        print("Capture")
    return [lr, fb, ud, yv]



while True:

    img = me.get_frame_read().frame
    # success, img = cap.read() 
    vals = getKeyboardInput() 
    me.send_rc_control(vals[0], vals[1], vals[2], vals[3])
    # print(vals)
    
    img = cv2.resize(img, (360, 240))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imshow("Image",img)

    # cv2.waitKey(1)
    # time.sleep(0.1)
    if cv2.waitKey(1) & 0xFF == ord('t'):
        break
# Release the webcam and close windows
print("End of Program...")
# cap.release()
cv2.destroyAllWindows()
