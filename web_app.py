from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import random

app = Flask(__name__)

class ActiveLiveness:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1, 
            min_detection_confidence=0.7, 
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        self.target_number = random.randint(1, 5)
        self.hold_time = 0
        self.required_hold_frames = 20
        self.challenges_passed = 0
        self.required_challenges = 5 # CHANGED TO 5 CHALLENGES
        self.liveness_verified = False

    def count_fingers(self, hand_landmarks, handedness):
        tip_ids = [4, 8, 12, 16, 20]
        fingers = []
        
        is_right = handedness.classification[0].label == 'Right'
        if is_right:
            fingers.append(1 if hand_landmarks.landmark[tip_ids[0]].x < hand_landmarks.landmark[tip_ids[0] - 1].x else 0)
        else:
            fingers.append(1 if hand_landmarks.landmark[tip_ids[0]].x > hand_landmarks.landmark[tip_ids[0] - 1].x else 0)

        for id in range(1, 5):
            if hand_landmarks.landmark[tip_ids[id]].y < hand_landmarks.landmark[tip_ids[id] - 2].y:
                fingers.append(1)
            else:
                fingers.append(0)
                
        return fingers.count(1)

def generate_frames():
    cap = cv2.VideoCapture(0)
    engine = ActiveLiveness()

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = engine.hands.process(rgb_frame)

        current_fingers = 0

        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                engine.mp_draw.draw_landmarks(frame, hand_landmarks, engine.mp_hands.HAND_CONNECTIONS)
                current_fingers = engine.count_fingers(hand_landmarks, handedness)

        # Draw UI over the frame stream
        if not engine.liveness_verified:
            cv2.putText(frame, f"Show me: {engine.target_number} fingers", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
            cv2.putText(frame, f"Passed: {engine.challenges_passed}/{engine.required_challenges}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

            if current_fingers == engine.target_number:
                engine.hold_time += 1
                bar_width = int((engine.hold_time / engine.required_hold_frames) * 300)
                cv2.rectangle(frame, (30, 110), (30 + bar_width, 130), (0, 255, 0), cv2.FILLED)
                
                if engine.hold_time >= engine.required_hold_frames:
                    engine.challenges_passed += 1
                    engine.hold_time = 0
                    if engine.challenges_passed >= engine.required_challenges:
                        engine.liveness_verified = True
                    else:
                        new_target = engine.target_number
                        while new_target == engine.target_number:
                            new_target = random.randint(1, 5)
                        engine.target_number = new_target
            else:
                engine.hold_time = 0
        else:
            cv2.putText(frame, "LIVENESS VERIFIED", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)

        # Encode the frame as a JPEG and stream it
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# --- Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    print("\n🌐 SERVER RUNNING! Open this link in your browser: http://127.0.0.1:5000\n")
    app.run(host='0.0.0.0', port=5000, debug=False)