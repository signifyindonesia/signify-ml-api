import cv2
import mediapipe as mp

class HandDetector:
    def __init__(self, max_hands=1, detection_confidence=0.7, draw_landmarks=False):
        self.max_hands = max_hands
        self.draw_landmarks = draw_landmarks
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,  # realtime mode
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils

    def find_hand(self, frame, return_landmarks=False):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        bbox = None
        landmarks = None

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                h, w, _ = frame.shape
                x_list = [lm.x * w for lm in handLms.landmark]
                y_list = [lm.y * h for lm in handLms.landmark]

                padding = 30
                x_min = max(int(min(x_list)) - padding, 0)
                y_min = max(int(min(y_list)) - padding, 0)
                x_max = min(int(max(x_list)) + padding, w)
                y_max = min(int(max(y_list)) + padding, h)

                if (x_max - x_min) < 40 or (y_max - y_min) < 40:
                    continue

                bbox = (x_min, y_min, x_max, y_max)

                if return_landmarks:
                    landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in handLms.landmark]

                if self.draw_landmarks:
                    self.mp_draw.draw_landmarks(frame, handLms, self.mp_hands.HAND_CONNECTIONS)

                break  # hanya satu tangan

        return (frame, bbox) if not return_landmarks else (frame, bbox, landmarks)
