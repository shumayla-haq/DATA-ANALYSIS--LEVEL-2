import cv2
model = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'weights.caffemodel')

image = cv2.imread('C:\\Users\\Nazim_HAQUE\\Downloads\\DATA-ANALYSIS -LEVEL-2-20231202T102458Z-001\\DATA-ANALYSIS -LEVEL-2\\SESSION 4 new\\Images\\input_image.jpg')

# Display the image in a window
#cv2.imshow('Window Name', image)

blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=True, crop=False)

model.setInput(blob)

detections = model.forward()

for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    if confidence > 0.5:  # Set a confidence threshold
        box = detections[0, 0, i, 3:7] * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
        (startX, startY, endX, endY) = box.astype(int)
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)

        cv2.imshow('Face Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()