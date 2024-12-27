from djitellopy import tello
from time import sleep

me = tello.Tello()
me.connect()
print("Battery: ",me.get_battery())

# me.takeoff()
# # me.send_rc_control(0, 0, 50, 0) # me.send_rc_control(lr, fb, ud, yv)
# sleep(2)
# me.send_rc_control(0, 50, 0, 0)
# sleep(2)
# me.send_rc_control(0, -50, 0, 0)
# # sleep(2)
# me.land()
