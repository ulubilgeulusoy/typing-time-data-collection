import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime

class HandTracker:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()
        self.mp_draw = mp.solutions.drawing_utils
        self.keyboard_detected = False
        self.keyboard_contour = None

    def detect_keyboard(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        largest_contour = max(contours, key=cv2.contourArea)
        
        if cv2.contourArea(largest_contour) < 5000:
            return None
        
        return largest_contour

    def track(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        if not self.keyboard_detected:
            self.keyboard_contour = self.detect_keyboard(frame)
            if self.keyboard_contour is not None:
                self.keyboard_detected = True
        
        hands_over_keyboard = False
        if results.multi_hand_landmarks:
            for landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(frame, landmarks, self.mp_hands.HAND_CONNECTIONS)
                if self.keyboard_detected:
                    for _, data_point in enumerate(landmarks.landmark):
                        pt = (int(data_point.x * frame.shape[1]), int(data_point.y * frame.shape[0]))
                        if cv2.pointPolygonTest(self.keyboard_contour, pt, False) > 0:
                            hands_over_keyboard = True
                            break
        
        if self.keyboard_detected:
            cv2.drawContours(frame, [self.keyboard_contour], -1, (0, 255, 0), 2)
        
        return frame, hands_over_keyboard

def main():
    #FOR RECORDED VIDEO FEED USAGE
    cap = cv2.VideoCapture('YOUR_FILE_PATH') 
    
    #FOR REAL TIME VIDEO FEED USAGE
    #cap = cv2.VideoCapture(0) 
    
    tracker = HandTracker()
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_count = 0
    hands_on_keyboard = False
    total_frames_processed = 0

    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break

        frame, hands_over_keyboard = tracker.track(frame)
        
        # Handle hand timer mechanism
        if hands_over_keyboard:
            frame_count += 1
            hands_on_keyboard = True
        elif hands_on_keyboard and not hands_over_keyboard:
            seconds = frame_count / fps
            timestamp = total_frames_processed / fps
            with open("DATA_FILE.txt", "a") as f:
                f.write(f"{seconds:.2f}; seconds at; {timestamp:.2f}; seconds; into the video \n")
            frame_count = 0
            hands_on_keyboard = False
        
        # Calculate and display the remaining time
        remaining_frames = total_frames - total_frames_processed
        remaining_time_seconds = remaining_frames / fps
        minutes, seconds = divmod(remaining_time_seconds, 60)
        timer_text = f"Time left: {int(minutes)}:{int(seconds):02d} min"
        cv2.putText(frame, timer_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow("Hand Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        total_frames_processed += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
