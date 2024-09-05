import cv2
import mediapipe as mp

# Initialize the pose estimation model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Open the default webcam
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Convert the frame to RGB for pose estimation
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform pose estimation
    results = pose.process(frame_rgb)

    # Draw the pose landmarks on the frame
    if results.pose_landmarks:
        mp_draw = mp.solutions.drawing_utils
        mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Print the detected pose landmarks
        print("Detected pose landmarks:")
        for landmark in results.pose_landmarks.landmark:
            print(f"x: {landmark.x}, y: {landmark.y}, z: {landmark.z}")

    # Display the resulting frame
    cv2.imshow('Pose Estimation', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()