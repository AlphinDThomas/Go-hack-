import cv2
import mediapipe as mp
import numpy as np
import random

class HeadPoseSequence:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Sequence Generation: 3 to 5 random turns
        self.sequence = [random.choice(["TURN LEFT", "TURN RIGHT"]) for _ in range(random.randint(3, 5))]
        self.current_step = 0
        
        # State Management
        self.hold_time = 0
        self.required_frames = 10 # Frames needed to confirm a turn
        self.awaiting_center = False # Forces user to look forward between turns
        
        # Security: Anomaly/Irregularity Tracking
        self.strikes = 0
        self.max_strikes = 2 # Fail the KYC if they mess up twice
        self.status = "ACTIVE" # Can be ACTIVE, PASSED, or FAILED

    def get_head_pose(self, frame, landmarks):
        img_h, img_w, _ = frame.shape
        face_3d = []
        face_2d = []

        key_indices = [1, 199, 33, 263, 61, 291]
        
        for idx in key_indices:
            lm = landmarks.landmark[idx]
            x, y = int(lm.x * img_w), int(lm.y * img_h)
            face_2d.append([x, y])
            face_3d.append([x, y, lm.z])

        face_2d = np.array(face_2d, dtype=np.float64)
        face_3d = np.array(face_3d, dtype=np.float64)

        focal_length = 1 * img_w
        cam_matrix = np.array([
            [focal_length, 0, img_h / 2],
            [0, focal_length, img_w / 2],
            [0, 0, 1]
        ])
        dist_matrix = np.zeros((4, 1), dtype=np.float64)

        success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
        rmat, _ = cv2.Rodrigues(rot_vec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

        pitch = angles[0] * 360
        yaw = angles[1] * 360
        
        return pitch, yaw

# --- Execution Loop ---
cap = cv2.VideoCapture(0)
engine = HeadPoseSequence()

print(f"Generated Challenge Sequence: {engine.sequence}")

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = engine.face_mesh.process(rgb_frame)

    yaw, pitch = 0, 0
    direction = "CENTER"

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            pitch, yaw = engine.get_head_pose(frame, face_landmarks)

            # Strict angle thresholds
            if yaw < -18: direction = "TURN LEFT"
            elif yaw > 18: direction = "TURN RIGHT"
            elif pitch < -15: direction = "LOOK DOWN"
            elif pitch > 15: direction = "LOOK UP"
            else: direction = "CENTER"

            # Draw nose pointer
            nose_x = int(face_landmarks.landmark[1].x * frame.shape[1])
            nose_y = int(face_landmarks.landmark[1].y * frame.shape[0])
            cv2.line(frame, (nose_x, nose_y), (int(nose_x + yaw * 2), int(nose_y - pitch * 2)), (255, 165, 0), 3)

    # --- Secure Sequence Logic ---
    if engine.status == "ACTIVE":
        # Draw Strike UI
        if engine.strikes > 0:
            cv2.putText(frame, f"WARNING: {engine.strikes}/{engine.max_strikes} STRIKES", (30, 130), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if engine.awaiting_center:
            cv2.putText(frame, "RETURN TO CENTER", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
            if direction == "CENTER":
                engine.hold_time += 1
                if engine.hold_time >= 5: # Require looking center for a split second
                    engine.awaiting_center = False
                    engine.hold_time = 0
            else:
                engine.hold_time = 0
        else:
            target = engine.sequence[engine.current_step]
            cv2.putText(frame, f"STEP {engine.current_step + 1}/{len(engine.sequence)}: {target}", (30, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)
            
            # Correct Action
            if direction == target:
                engine.hold_time += 1
                cv2.rectangle(frame, (30, 80), (30 + (engine.hold_time * 20), 100), (0, 255, 0), -1)
                
                if engine.hold_time >= engine.required_frames:
                    engine.current_step += 1
                    engine.hold_time = 0
                    if engine.current_step >= len(engine.sequence):
                        engine.status = "PASSED"
                    else:
                        engine.awaiting_center = True # Force reset for next step
                        
            # Irregularity Detected (Looking the wrong way)
            elif direction != "CENTER" and direction in ["TURN LEFT", "TURN RIGHT"] and direction != target:
                engine.hold_time += 1
                cv2.rectangle(frame, (30, 80), (30 + (engine.hold_time * 20), 100), (0, 0, 255), -1) # Red progress bar
                
                if engine.hold_time >= engine.required_frames:
                    engine.strikes += 1
                    engine.hold_time = 0
                    engine.awaiting_center = True # Force them back to center
                    
                    if engine.strikes >= engine.max_strikes:
                        engine.status = "FAILED"
            else:
                engine.hold_time = 0 # Reset if they don't commit to a direction

    elif engine.status == "PASSED":
        cv2.putText(frame, "LIVENESS SEQUENCE VERIFIED", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    
    elif engine.status == "FAILED":
        cv2.putText(frame, "KYC FAILED: ANOMALY DETECTED", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv2.imshow('Pulse-Guard: Cognitive Sequence', frame)
    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()