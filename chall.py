import cv2
import mediapipe as mp
import random
import time

class ActiveLiveness:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1, 
            min_detection_confidence=0.7, 
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Game State Variables
        self.target_number = random.randint(1, 5)
        self.hold_time = 0
        self.required_hold_frames = 20 # Must hold the gesture for ~1 second
        self.challenges_passed = 0
        self.required_challenges = 3
        self.liveness_verified = False

    def count_fingers(self, hand_landmarks, handedness):
        tip_ids = [4, 8, 12, 16, 20] # Thumb, Index, Middle, Ring, Pinky
        fingers = []
        
        # 1. Thumb Logic (Depends on Left or Right hand)
        is_right = handedness.classification[0].label == 'Right'
        if is_right:
            # Right hand thumb opens to the left (lower X value)
            fingers.append(1 if hand_landmarks.landmark[tip_ids[0]].x < hand_landmarks.landmark[tip_ids[0] - 1].x else 0)
        else:
            # Left hand thumb opens to the right (higher X value)
            fingers.append(1 if hand_landmarks.landmark[tip_ids[0]].x > hand_landmarks.landmark[tip_ids[0] - 1].x else 0)

        # 2. Four Fingers Logic (Tip vs Knuckle on Y axis)
        for id in range(1, 5):
            if hand_landmarks.landmark[tip_ids[id]].y < hand_landmarks.landmark[tip_ids[id] - 2].y:
                fingers.append(1)
            else:
                fingers.append(0)
                
        return fingers.count(1)

# --- Execution Loop ---
cap = cv2.VideoCapture(0)
engine = ActiveLiveness()

print("Starting Active Challenge Liveness...")

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    # Flip frame horizontally for a mirror effect, then convert color
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = engine.hands.process(rgb_frame)

    current_fingers = 0

    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            # Draw the skeleton on the hand
            engine.mp_draw.draw_landmarks(frame, hand_landmarks, engine.mp_hands.HAND_CONNECTIONS)
            
            # Get finger count
            current_fingers = engine.count_fingers(hand_landmarks, handedness)

    # --- Liveness State Machine ---
    if not engine.liveness_verified:
        # UI: Show instructions
        cv2.putText(frame, f"Show me: {engine.target_number} fingers", (30, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        cv2.putText(frame, f"Challenges passed: {engine.challenges_passed}/{engine.required_challenges}", (30, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        if current_fingers == engine.target_number:
            engine.hold_time += 1
            # Draw Progress Bar
            bar_width = int((engine.hold_time / engine.required_hold_frames) * 300)
            cv2.rectangle(frame, (30, 110), (30 + bar_width, 130), (0, 255, 0), cv2.FILLED)
            
            # If gesture held long enough, pass the challenge
            if engine.hold_time >= engine.required_hold_frames:
                engine.challenges_passed += 1
                engine.hold_time = 0
                if engine.challenges_passed >= engine.required_challenges:
                    engine.liveness_verified = True
                else:
                    # Generate a new random number that isn't the current one
                    new_target = engine.target_number
                    while new_target == engine.target_number:
                        new_target = random.randint(1, 5)
                    engine.target_number = new_target
        else:
            engine.hold_time = 0 # Reset if they drop the gesture
    else:
        # Success Screen
        cv2.putText(frame, "LIVENESS VERIFIED", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)
        cv2.putText(frame, "ACCESS GRANTED", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

    # Developer UI: Always show current detected fingers
    cv2.putText(frame, f"Detected: {current_fingers}", (450, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('Pulse-Guard: Active Response', frame)
    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()