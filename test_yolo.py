from ultralytics import YOLO
import cv2

model = YOLO('models/yolov8n.pt')
results = model('Images/1.jpg', show=True) 

cv2.waitKey(0)