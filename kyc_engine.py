import cv2
import mediapipe as mp
import numpy as np
import random
import collections

class KYCEngine:
    def __init__(self):
        self.mp_face = mp.solutions.face_mesh
        self.face_mesh = self.mp_face.FaceMesh(refine_landmarks=True)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1)
        self.mp_draw = mp.solutions.drawing_utils
        
        self.state = 'ID_SCAN'
        
        self.pulse_buffer = collections.deque(maxlen=150)
        self.pulse_times = collections.deque(maxlen=150)
        
        self.gesture_target = random.randint(1, 5)
        self.gesture_count = 0
        self.gesture_goal = 5
        
        # Head Pose sequence generator (3 random turns)
        self.head_sequence = [random.choice(["LEFT", "RIGHT"]) for _ in range(3)]
        self.head_step = 0
        self.awaiting_center = False
        
        self.strikes = 0
        self.hold_timer = 0

    def process_frame(self, frame):
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        if self.state == 'ID_SCAN': return self._stage_id_scan(frame, w, h)
        elif self.state == 'PULSE': return self._stage_pulse(frame, rgb)
        elif self.state == 'GESTURE': return self._stage_gesture(frame, rgb)
        elif self.state == 'HEAD_POSE': return self._stage_head(frame, rgb, w, h)
        elif self.state == 'SUCCESS': return self._stage_success(frame, w, h)
        elif self.state == 'FAILED': return self._stage_failed(frame, w, h)
        
        return frame

    def _stage_id_scan(self, frame, w, h):
        cv2.rectangle(frame, (150, 100), (w-150, h-100), (255, 255, 255), 2)
        cv2.putText(frame, "HOLD ID CARD INSIDE BOX", (170, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(frame, "CLICK 'CONFIRM ID' ON DASHBOARD", (170, h-60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        return frame

    def _stage_pulse(self, frame, rgb):
        results = self.face_mesh.process(rgb)
        if results.multi_face_landmarks:
            cv2.putText(frame, "ANALYZING BIOMETRIC PULSE...", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            self.hold_timer += 1
            cv2.rectangle(frame, (30, 70), (30 + int((self.hold_timer/60)*300), 85), (0, 255, 0), cv2.FILLED)
            if self.hold_timer > 60:
                self.state = 'GESTURE'
                self.hold_timer = 0
        return frame

    def _stage_gesture(self, frame, rgb):
        results = self.hands.process(rgb)
        cv2.putText(frame, f"CHALLENGE: SHOW {self.gesture_target} FINGERS", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
        cv2.putText(frame, f"Progress: {self.gesture_count}/{self.gesture_goal}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        
        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                tip_ids = [4, 8, 12, 16, 20]
                fingers = []
                is_right = handedness.classification[0].label == 'Right'
                if is_right: fingers.append(1 if hand_landmarks.landmark[tip_ids[0]].x < hand_landmarks.landmark[tip_ids[0] - 1].x else 0)
                else: fingers.append(1 if hand_landmarks.landmark[tip_ids[0]].x > hand_landmarks.landmark[tip_ids[0] - 1].x else 0)

                for id in range(1, 5):
                    if hand_landmarks.landmark[tip_ids[id]].y < hand_landmarks.landmark[tip_ids[id] - 2].y: fingers.append(1)
                    else: fingers.append(0)
                        
                current_fingers = fingers.count(1)
                
                if current_fingers == self.gesture_target:
                    self.hold_timer += 1
                    cv2.rectangle(frame, (30, 110), (30 + (self.hold_timer * 15), 130), (0, 255, 0), cv2.FILLED)
                    if self.hold_timer > 20:
                        self.gesture_count += 1
                        self.hold_timer = 0
                        if self.gesture_count >= self.gesture_goal:
                            self.state = 'HEAD_POSE'
                        else:
                            new_target = self.gesture_target
                            while new_target == self.gesture_target: new_target = random.randint(1, 5)
                            self.gesture_target = new_target
                else: self.hold_timer = 0
        return frame

    def _stage_head(self, frame, rgb, w, h):
        results = self.face_mesh.process(rgb)
        yaw, pitch = 0, 0
        direction = "CENTER"

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # 3D Math for Head Rotation
                face_2d = []
                face_3d = []
                for idx in [1, 199, 33, 263, 61, 291]:
                    lm = face_landmarks.landmark[idx]
                    x, y = int(lm.x * w), int(lm.y * h)
                    face_2d.append([x, y])
                    face_3d.append([x, y, lm.z])
                
                face_2d = np.array(face_2d, dtype=np.float64)
                face_3d = np.array(face_3d, dtype=np.float64)
                cam_matrix = np.array([[w, 0, w/2], [0, w, h/2], [0, 0, 1]], dtype=np.float64)
                success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, np.zeros((4, 1), dtype=np.float64))
                rmat, _ = cv2.Rodrigues(rot_vec)
                angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
                pitch, yaw = angles[0] * 360, angles[1] * 360

                if yaw < -18: direction = "LEFT"
                elif yaw > 18: direction = "RIGHT"
                else: direction = "CENTER"
                
                nx, ny = int(face_landmarks.landmark[1].x * w), int(face_landmarks.landmark[1].y * h)
                cv2.line(frame, (nx, ny), (int(nx + yaw * 2), int(ny - pitch * 2)), (255, 165, 0), 3)

        # Sequence Logic
        if self.strikes > 0:
            cv2.putText(frame, f"WARNING: {self.strikes}/2 STRIKES", (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if self.awaiting_center:
            cv2.putText(frame, "RETURN TO CENTER", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
            if direction == "CENTER":
                self.hold_timer += 1
                if self.hold_timer >= 5:
                    self.awaiting_center = False
                    self.hold_timer = 0
            else: self.hold_timer = 0
        else:
            target = self.head_sequence[self.head_step]
            cv2.putText(frame, f"STEP {self.head_step + 1}/{len(self.head_sequence)}: TURN {target}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)
            
            if direction == target:
                self.hold_timer += 1
                cv2.rectangle(frame, (30, 80), (30 + (self.hold_timer * 20), 100), (0, 255, 0), -1)
                if self.hold_timer >= 10:
                    self.head_step += 1
                    self.hold_timer = 0
                    if self.head_step >= len(self.head_sequence):
                        self.state = 'SUCCESS'
                    else:
                        self.awaiting_center = True
            elif direction != "CENTER" and direction in ["LEFT", "RIGHT"] and direction != target:
                self.hold_timer += 1
                cv2.rectangle(frame, (30, 80), (30 + (self.hold_timer * 20), 100), (0, 0, 255), -1)
                if self.hold_timer >= 10:
                    self.strikes += 1
                    self.hold_timer = 0
                    self.awaiting_center = True
                    if self.strikes >= 2:
                        self.state = 'FAILED'
            else:
                self.hold_timer = 0
                
        return frame

    def _stage_success(self, frame, w, h):
        cv2.rectangle(frame, (0,0), (w,h), (0, 255, 0), 10)
        cv2.putText(frame, "IDENTITY VERIFIED", (int(w/2)-150, int(h/2)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 4)
        return frame

    def _stage_failed(self, frame, w, h):
        cv2.rectangle(frame, (0,0), (w,h), (0, 0, 255), 10)
        cv2.putText(frame, "KYC FAILED: ANOMALY DETECTED", (int(w/2)-250, int(h/2)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 4)
        return frame
    def reset(self):
        """Resets the entire state machine for a new user session."""
        self.state = 'ID_SCAN'
        self.pulse_buffer.clear()
        self.pulse_times.clear()
        
        self.gesture_target = random.randint(1, 5)
        self.gesture_count = 0
        
        self.head_sequence = [random.choice(["LEFT", "RIGHT"]) for _ in range(3)]
        self.head_step = 0
        self.awaiting_center = False
        
        self.strikes = 0
        self.hold_timer = 0