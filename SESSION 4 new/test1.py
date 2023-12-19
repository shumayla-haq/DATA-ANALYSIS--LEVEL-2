import cv2
face_cascade = cv2.CascadeClassifier('C:\\Users\\Nazim_HAQUE\\Downloads\\DATA-ANALYSIS -LEVEL-2-20231202T102458Z-001\\DATA-ANALYSIS -LEVEL-2\\SESSION 4 new\\haarcascade_frontalface_default.xml')
image = cv2.imread('C:\\Users\\Nazim_HAQUE\\Downloads\\DATA-ANALYSIS -LEVEL-2-20231202T102458Z-001\\DATA-ANALYSIS -LEVEL-2\\SESSION 4 new\\Images\\input_image.jpg')

# Display the image in a window
#cv2.imshow('Window Name', image)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('Face Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()