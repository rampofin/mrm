import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from model import CNN 

load_from_sys = True

if load_from_sys:
    hsv_value = np.load('hsv_value.npy')
else:
    hsv_value = np.array([[0, 0, 0], [179, 255, 255]])

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

kernel = np.ones((5, 5), np.uint8)
canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
x1, y1 = 0, 0
noise_thresh = 800

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN().to(device)
model.load_state_dict(torch.load('model.pth', map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(),
    transforms.ToTensor()
])

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_range = hsv_value[0]
    upper_range = hsv_value[1]

    mask = cv2.inRange(hsv, lower_range, upper_range)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours and cv2.contourArea(max(contours, key=cv2.contourArea)) > noise_thresh:
        c = max(contours, key=cv2.contourArea)
        x2, y2, w, h = cv2.boundingRect(c)

        if x1 == 0 and y1 == 0:
            x1, y1 = x2, y2
        else:
            canvas = cv2.line(canvas, (x1, y1), (x2, y2), [0, 255, 0], 4)

        x1, y1 = x2, y2
    else:
        x1, y1 = 0, 0

    frame = cv2.add(canvas, frame)
    stacked = np.hstack((canvas, frame))
    cv2.imshow('Screen_Pen', cv2.resize(stacked, None, fx=0.6, fy=0.6))

    key = cv2.waitKey(1)
    if key == 10:  #  key
        gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        gray_canvas = cv2.resize(gray_canvas, (28, 28))
        tensor_canvas = transform(gray_canvas).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(tensor_canvas)
            prediction = output.argmax(dim=1, keepdim=True).item()
            print(f'Prediction: {prediction}')
        
        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)

    if key == ord('c'):
        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)

    if key == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()
