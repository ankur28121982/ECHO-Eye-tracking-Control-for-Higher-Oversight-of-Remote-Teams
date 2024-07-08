import cv2
import dlib
import numpy as np
import datetime
import time

# Initialize Dlib's face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialize OpenCV's video capture
cap = cv2.VideoCapture(0)

# Open a file to log eye movement data
log_file = open("eye_movement_log.txt", "w")

# User name
user_name = "YourUserName"  # Replace with the actual user name

# Function to calculate the degree of movement
def calculate_movement_degree(current_eye_center, previous_eye_center):
    return np.linalg.norm(current_eye_center - previous_eye_center)

# Variables to track eye movement and log intervals
movement_threshold = 20  # Adjust this threshold as needed
log_interval = 15  # Log data every 15 seconds
start_time = time.time()
previous_eye_center = None
user_warning = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        # Example: tracking eye movement (you may need to adjust this based on your requirements)
        left_eye = np.array([(landmarks.part(36).x, landmarks.part(36).y),
                             (landmarks.part(37).x, landmarks.part(37).y),
                             (landmarks.part(38).x, landmarks.part(38).y),
                             (landmarks.part(39).x, landmarks.part(39).y),
                             (landmarks.part(40).x, landmarks.part(40).y),
                             (landmarks.part(41).x, landmarks.part(41).y)], np.int32)

        # Calculate the center of the eye
        current_eye_center = np.mean(left_eye, axis=0)

        if previous_eye_center is not None:
            movement_degree = calculate_movement_degree(current_eye_center, previous_eye_center)
            
            # Check if the movement exceeds the threshold
            if movement_degree > movement_threshold:
                user_warning = True
            else:
                user_warning = False

            # Log the eye movement data with timestamp
            if time.time() - start_time >= log_interval:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                # Format the movement_degree to two decimal places
                log_file.write(f"{timestamp}, {user_name}, {current_eye_center[0]:.2f}, {current_eye_center[1]:.2f}, {movement_degree:.2f}\n")
                start_time = time.time()
        else:
            movement_degree = 0

        previous_eye_center = current_eye_center

        # Display text on the frame
        cv2.putText(frame, f"Eye Movement: {movement_degree:.2f}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if user_warning:
        warning_text = "Dear User, please remain concentrated"
        # Get text size
        text_size, _ = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        text_x = (frame.shape[1] - text_size[0]) // 2
        text_y = (frame.shape[0] + text_size[1]) // 2
        cv2.putText(frame, warning_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Eye Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
log_file.close()
