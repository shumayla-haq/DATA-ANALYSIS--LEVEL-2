'''import cv2
face_cascade = cv2.CascadeClassifier('C:\\Users\\OMOTEC041\\Downloads\\haarcascade_frontalface_default.xml')
image = cv2.imread("C:\\Users\\OMOTEC041\\Pictures\\id-photo2.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
cv2.imshow('Face Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

"C:\\Users\\OMOTEC041\\Downloads\\deploy (1).prototxt","C:\\Users\\OMOTEC041\\Downloads\\weights.caffemodel"
"C:\\Users\\OMOTEC041\\Pictures\\id-photo2.jpg"


import numpy as np
import cv2

# Set the values directly in the notebook
args = {
    "video": "",  # Set the path to the video file or leave it empty for camera stream
    "prototxt": "C:\\users\\Nazim_HAQUE\\Downloads\\DATA-ANALYSIS -LEVEL-2-20231202T102458Z-001\\DATA-ANALYSIS -LEVEL-2\\SESSION 4 new\\MobileNetSSD_deploy.prototxt",
    "weights": "C:\\users\\Nazim_HAQUE\\Downloads\\DATA-ANALYSIS -LEVEL-2-20231202T102458Z-001\\DATA-ANALYSIS -LEVEL-2\\SESSION 4 new\\MobileNetSSD_deploy.caffemodel",
    "thr": 0.2
}

# Labels of Network.
classNames = {0: 'background', 1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
              5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair',
              10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse',
              14: 'motorbike', 15: 'person', 16: 'pottedplant',
              17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor'}

# Open video file or capture device.
if args["video"]:
    cap = cv2.VideoCapture(args["video"])
else:
    cap = cv2.VideoCapture(0)

# Load the Caffe model
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["weights"])

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame_resized = cv2.resize(frame, (300, 300))  # resize frame for prediction

    # MobileNet requires fixed dimensions for input image(s)
    blob = cv2.dnn.blobFromImage(frame_resized, 0.007843, (300, 300), (127.5, 127.5, 127.5), False)
    net.setInput(blob)
    detections = net.forward()

    cols = frame_resized.shape[1]
    rows = frame_resized.shape[0]

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > args["thr"]:
            class_id = int(detections[0, 0, i, 1])

            xLeftBottom = int(detections[0, 0, i, 3] * cols)
            yLeftBottom = int(detections[0, 0, i, 4] * rows)
            xRightTop = int(detections[0, 0, i, 5] * cols)
            yRightTop = int(detections[0, 0, i, 6] * rows)

            heightFactor = frame.shape[0] / 300.0
            widthFactor = frame.shape[1] / 300.0
            xLeftBottom = int(widthFactor * xLeftBottom)
            yLeftBottom = int(heightFactor * yLeftBottom)
            xRightTop = int(widthFactor * xRightTop)
            yRightTop = int(heightFactor * yRightTop)

            cv2.rectangle(frame, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop), (0, 255, 0))

            if class_id in classNames:
                label = classNames[class_id] + ": " + str(confidence)
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                yLeftBottom = max(yLeftBottom, labelSize[1])
                cv2.rectangle(frame, (xLeftBottom, yLeftBottom - labelSize[1]),
                              (xLeftBottom + labelSize[0], yLeftBottom + baseLine),
                              (255, 255, 255), cv2.FILLED)
                cv2.putText(frame, label, (xLeftBottom, yLeftBottom),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

                print(label)

    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) >= 0:  # Break with ESC
        break

# Release the video capture object and destroy all windows
cap.release()
cv2.destroyAllWindows()

