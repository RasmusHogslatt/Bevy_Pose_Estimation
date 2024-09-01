import cv2  # Ensure cv2 is imported correctly
import mediapipe as mp
import numpy as np
import pickle
import os
import struct
import sys

def process_frame(frame):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=True, min_detection_confidence=0.5)
    
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    if not results.pose_world_landmarks:
        return []
    
    return [(lm.x, lm.y, lm.z) for lm in results.pose_world_landmarks.landmark]

def main():
    # Attempt to open the pipes with error handling
    try:
        frame_pipe = open("frame_pipe", "rb")
        points_pipe = open("points_pipe", "wb")
    except Exception as e:
        print(f"Failed to open pipes: {e}", file=sys.stderr)
        return

    frame_width = 640  # Adjust based on your camera resolution
    frame_height = 480
    
    try:
        while True:
            # Read frame data from pipe
            frame_data = frame_pipe.read(frame_width * frame_height * 3)  # Assuming 3 channels (BGR)
            if not frame_data:
                break

            # Convert frame data to numpy array
            frame = np.frombuffer(frame_data, dtype=np.uint8).reshape((frame_height, frame_width, 3))
            
            # Process frame
            points_3d = process_frame(frame)
            
            # Serialize 3D points using pickle
            points_data = pickle.dumps(points_3d)
            
            # Send the size of the data first, then the data itself
            points_pipe.write(struct.pack('I', len(points_data)))
            points_pipe.write(points_data)
            points_pipe.flush()  # Ensure data is sent immediately
            
    except Exception as e:
        print(f"An error occurred during processing: {e}", file=sys.stderr)
    
    finally:
        # Ensure the pipes are properly closed
        frame_pipe.close()
        points_pipe.close()

if __name__ == "__main__":
    main()
