import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from scipy import stats

class HandGestureRecognition:
    def __init__(self, camera_index=0, min_detection_confidence=0.5, model_complexity=1):
        self.camera_index = camera_index
        self.cap = cv2.VideoCapture(camera_index)
        self.min_detection_confidence = min_detection_confidence
        self.model_complexity = model_complexity
        self.mp_hands = mp.solutions.hands.Hands(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5,
            model_complexity=model_complexity
        )
        self.finger_counts = deque(maxlen=10)
        self.gesture_dict = {
            "Fist Bump!!!": "Play/Pause video",
            "Don't point finger at me": "Change slide",
            "Peace!!!": "Change slide"
        }

    def get_finger_info(self, landmarks, handedness):
        finger_tip_indices = [mp.solutions.hands.HandLandmark.THUMB_TIP,
                              mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP,
                              mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP,
                              mp.solutions.hands.HandLandmark.RING_FINGER_TIP,
                              mp.solutions.hands.HandLandmark.PINKY_TIP]
        finger_middle_indices = [mp.solutions.hands.HandLandmark.THUMB_IP,
                                 mp.solutions.hands.HandLandmark.INDEX_FINGER_PIP,
                                 mp.solutions.hands.HandLandmark.MIDDLE_FINGER_PIP,
                                 mp.solutions.hands.HandLandmark.RING_FINGER_PIP,
                                 mp.solutions.hands.HandLandmark.PINKY_PIP]

        finger_tips = [landmarks[i] for i in finger_tip_indices]
        finger_middles = [landmarks[i] for i in finger_middle_indices]

        extended_fingers = [tip.y < middle.y for tip, middle in zip(finger_tips, finger_middles)]

        finger_count = sum(extended_fingers)

        if finger_count == 0:
            gesture = "Fist Bump!!!"
        elif finger_count == 1 and extended_fingers[1]:
            gesture = "Don't point finger at me"
        elif finger_count == 2 and extended_fingers[1] and extended_fingers[2]:
            gesture = "Peace!!!"
        else:
            gesture = None

        return finger_count, gesture, handedness

    def run(self):
        while True:
            success, image = self.cap.read()
            if not success:
                print("Failed to capture video frame.")
                break

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.mp_hands.process(image_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    h, w, _ = image.shape
                    x_min = int(min(landmark.x for landmark in hand_landmarks.landmark) * w)
                    y_min = int(min(landmark.y for landmark in hand_landmarks.landmark) * h)
                    x_max = int(max(landmark.x for landmark in hand_landmarks.landmark) * w)
                    y_max = int(max(landmark.y for landmark in hand_landmarks.landmark) * h)

                    finger_count, gesture, handedness = self.get_finger_info(hand_landmarks.landmark, handedness.classification[0].label)

                    self.finger_counts.append(finger_count)
                    mode = stats.mode(self.finger_counts)[0]
                    finger_count = mode[0] if np.ndim(mode) > 0 else mode

                    if gesture is not None:
                        print(gesture)
                        if gesture in self.gesture_dict:
                            print(f"Action: {self.gesture_dict[gesture]}")

                    cv2.putText(image, f"Fingers: {finger_count} ({handedness})", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                    if gesture is not None:
                        cv2.putText(image, f"Gesture: {gesture}", (x_min, y_min - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    mp.solutions.drawing_utils.draw_landmarks(image, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

            cv2.imshow('Frame', image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    gesture_recognizer = HandGestureRecognition()
    gesture_recognizer.run()
