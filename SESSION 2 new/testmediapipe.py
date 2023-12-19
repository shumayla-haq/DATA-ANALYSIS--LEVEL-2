import cv2
import mediapipe as mp

# Load the Mediapipe hand landmark model
mp_hands = mp.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Initialize the video capture object
cap = cv2.VideoCapture(0)

while True:
    # Read a new frame from the video capture object
    ret, frame = cap.read()

    # Check if the frame is empty
    if not ret:
        break

    # Convert the color space from BGR to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect the hand landmarks in the current frame
    results = mp_hands.process(frame)

    # Draw the hand landmarks on the current frame
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp.drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the current frame in a window
    cv2.imshow('Hand Landmarks', frame)

    # Check for a key event and exit if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video capture object and destroy all windows
cap.release()
cv2.destroyAllWindows()
