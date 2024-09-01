import cv2
import mediapipe as mp
import time

def estimate(frame, pose):
    results = pose.process(frame)
    if results.pose_landmarks:
        return [(lm.x, lm.y) for lm in results.pose_landmarks.landmark]
    return []

def test_estimate_function_webcam():
    cap = cv2.VideoCapture(0)
    
    # MediaPipe Pose Configuration
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1  # 0 for fastest, 2 for most accurate
    )
    
    frame_count = 0
    process_this_frame = True
    prev_frame_time = 0
    new_frame_time = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        frame_count += 1
        
        # Resize frame
        frame = cv2.resize(frame, (640, 480))
        
        if process_this_frame:
            # Convert the BGR image to RGB
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Apply the estimate function
            keypoints = estimate(image_rgb, pose)

            # Draw the keypoints on the frame
            for point in keypoints:
                x, y = int(point[0] * frame.shape[1]), int(point[1] * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            # Display the number of keypoints
            cv2.putText(frame, f"Keypoints: {len(keypoints)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Calculate and display FPS
        new_frame_time = time.time()
        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('Pose Estimation', frame)

        # Process every 3rd frame
        process_this_frame = frame_count % 3 == 0

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_estimate_function_webcam()