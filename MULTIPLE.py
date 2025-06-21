from ultralytics import YOLO
import cvzone
import cv2
import math
import threading
import time
import serial

ser = serial.Serial('COM10',9600)
class VideoStream:
    def __init__(self, src=0, width=480, height=360):
        # Initialize the video stream (webcam or camera source)
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.ret, self.frame = self.stream.read()
        self.stopped = False

    def start(self):
        # Start the thread for updating frames
        threading.Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while not self.stopped:
            if not self.ret:
                self.stop()
            else:
                self.ret, self.frame = self.stream.read()

    def read(self):
        # Return the current frame
        return self.frame

    def stop(self):
        # Stop the video stream
        self.stopped = True
        self.stream.release()


# Load the two YOLO models
Con_model = YOLO('CONSTRUCTION.pt')
shoes_model = YOLO('SHOE.pt')
gloves_model = YOLO('GLOVES.pt')


construction_classnames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone', 'Safety Vest', 'machinery', 'vehicle']
shoes_classnames = ['no-safety-shoes','safety-shoes']
gloves_classnames = ['bare hand','glove']

vs = VideoStream(src=0, width=720, height=640).start()

# Frame skipping parameter
frame_skip = 5
frame_count = 0

# Main loop
while True:
    frame = vs.read()  # Read the current frame from the video stream

    if frame is None:
        break

    frame_count += 1

    # Skip frames to reduce lag
    if frame_count % frame_skip != 0:
        continue

    construction_results = Con_model(frame, stream=True)
    shoes_results = shoes_model(frame, stream=True)
    gloves_results = gloves_model(frame, stream=True)

    for info in construction_results:
        boxes = info.boxes
        for box in boxes:
            confidence = box.conf[0]
            confidence = math.ceil(confidence * 100)  # Convert to percentage
            Class = int(box.cls[0])  # Get class index
            if confidence > 30:  # Confidence threshold
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)  # Red rectangle for vehicles
                cvzone.putTextRect(frame, f'{construction_classnames[Class]} {confidence}%',
                                   [x1 + 8, y1 + 100], scale=1.5, thickness=2)
                print('construction detected:', construction_classnames[Class], 'Confidence:', confidence)

                if construction_classnames[Class] == "Hardhat":
                    print("Hardhat")
                    b = b'C'
                    ser.write(b)
                elif construction_classnames[Class] == "Mask":
                    print("Mask")
                    b = b'D'
                    ser.write(b)
                elif construction_classnames[Class] == "NO-Hardhat":
                    print("NO-Hardhat")
                    b = b'E'
                    ser.write(b)
                elif construction_classnames[Class] == "NO-Mask":
                    print("NO-Mask")
                    b = b'F'
                    ser.write(b)
                elif construction_classnames[Class] == "NO-Safety Vest":
                    print("NO-Safety Vest")
                    b = b'G'
                    ser.write(b)
                elif construction_classnames[Class] == "Safety Vest":
                    print("Safety Vest")
                    b = b'H'
                    ser.write(b)

    for info in shoes_results:
        boxes = info.boxes
        for box in boxes:
            confidence = box.conf[0]
            confidence = math.ceil(confidence * 100)  # Convert to percentage
            Class = int(box.cls[0])  # Get class index
            if confidence > 30:  # Confidence threshold
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)  # Green rectangle for vegetables
                cvzone.putTextRect(frame, f'{shoes_classnames[Class]} {confidence}%',
                                   [x1 + 8, y1 + 100], scale=1.5, thickness=2)
                print('shoes detected:', shoes_classnames[Class], 'Confidence:', confidence)


    for info in gloves_results:
        boxes = info.boxes
        for box in boxes:
            confidence = box.conf[0]
            confidence = math.ceil(confidence * 100)  # Convert to percentage
            Class = int(box.cls[0])  # Get class index
            if confidence > 30:  # Confidence threshold
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)  # Green rectangle for vegetables
                cvzone.putTextRect(frame, f'{gloves_classnames[Class]} {confidence}%',
                                   [x1 + 8, y1 + 100], scale=1.5, thickness=2)
                print('gloves detected:', gloves_classnames[Class], 'Confidence:', confidence)

    # Display the frame with bounding boxes from both models
    cv2.imshow('frame', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Stop the video stream and release resources
vs.stop()
cv2.destroyAllWindows()
