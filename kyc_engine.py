import cv2
import mediapipe as mp
import numpy as np
import random
import collections
import face_recognition


class KYCEngine:
    def __init__(self):
        self.mp_face = mp.solutions.face_mesh
        self.face_mesh = self.mp_face.FaceMesh(refine_landmarks=True)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1)
        self.mp_draw = mp.solutions.drawing_utils

        self.state = 'ID_SCAN'

        # ── Face Registration ──────────────────────────────────────
        self.registered_face_encoding = None   # numpy array set by register_face()
        self.face_match_confidence = 0.0       # 0–100, surfaced in /get_status
        self.face_match_status = "NO_USER_REGISTERED"
        # MATCH | NO_MATCH | SCANNING | NO_FACE_DETECTED | NO_USER_REGISTERED

        # Frame-skip so face_recognition doesn't block the video stream
        self._frame_count = 0
        self._FACE_CHECK_EVERY = 4             # run recognition every N frames

        # ── Pulse ──────────────────────────────────────────────────
        self.pulse_buffer = collections.deque(maxlen=150)
        self.pulse_times = collections.deque(maxlen=150)

        # ── Gesture ────────────────────────────────────────────────
        self.gesture_target = random.randint(1, 5)
        self.gesture_count = 0
        self.gesture_goal = 5

        # ── Head Pose ──────────────────────────────────────────────
        self.head_sequence = [random.choice(["LEFT", "RIGHT"]) for _ in range(3)]
        self.head_step = 0
        self.awaiting_center = False

        # ── Session ────────────────────────────────────────────────
        self.strikes = 0
        self.hold_timer = 0

    # ──────────────────────────────────────────────────────────────
    # Public: register a face from raw image bytes
    # ──────────────────────────────────────────────────────────────
    def register_face(self, image_bytes: bytes) -> tuple[bool, str]:
        """
        Decode image bytes, extract face encoding, store it.
        Returns (success: bool, message: str).
        """
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return False, "Could not decode image. Upload a JPG or PNG."

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(rgb, model="hog")
        if not locations:
            return False, "No face detected in the uploaded photo. Try better lighting or a clearer shot."

        if len(locations) > 1:
            return False, f"{len(locations)} faces found. Upload a photo with exactly one face."

        encodings = face_recognition.face_encodings(rgb, locations)
        self.registered_face_encoding = encodings[0]
        self.face_match_status = "SCANNING"
        self.face_match_confidence = 0.0
        return True, "Face registered successfully."

    # ──────────────────────────────────────────────────────────────
    # Main frame dispatcher
    # ──────────────────────────────────────────────────────────────
    def process_frame(self, frame):
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self.state == 'ID_SCAN':   return self._stage_id_scan(frame, rgb, w, h)
        elif self.state == 'PULSE':   return self._stage_pulse(frame, rgb)
        elif self.state == 'GESTURE': return self._stage_gesture(frame, rgb)
        elif self.state == 'HEAD_POSE': return self._stage_head(frame, rgb, w, h)
        elif self.state == 'SUCCESS': return self._stage_success(frame, w, h)
        elif self.state == 'FAILED':  return self._stage_failed(frame, w, h)
        return frame

    # ──────────────────────────────────────────────────────────────
    # Stage 1 — ID Scan + Real-Time Face Match
    # ──────────────────────────────────────────────────────────────
    def _stage_id_scan(self, frame, rgb, w, h):
        # Guide rectangle (where the user should position their face)
        box_x1, box_y1, box_x2, box_y2 = 150, 80, w - 150, h - 80
        cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (255, 255, 255), 2)

        if self.registered_face_encoding is None:
            # No user enrolled yet — prompt operator
            cv2.putText(frame, "NO USER REGISTERED", (box_x1 + 10, box_y1 - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 100, 255), 2)
            cv2.putText(frame, "Upload a registered face photo on the dashboard first",
                        (box_x1 + 10, h - 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 165, 255), 1)
            self.face_match_status = "NO_USER_REGISTERED"
            self.face_match_confidence = 0.0
            return frame

        # ── Run face recognition every N frames ──────────────────
        self._frame_count += 1
        if self._frame_count % self._FACE_CHECK_EVERY == 0:
            # Downsample for speed
            small = cv2.resize(rgb, (0, 0), fx=0.5, fy=0.5)
            locations = face_recognition.face_locations(small, model="hog")

            if not locations:
                self.face_match_status = "NO_FACE_DETECTED"
                self.face_match_confidence = 0.0
            else:
                # Scale locations back up
                locations_full = [(t*2, r*2, b*2, l*2) for (t, r, b, l) in locations]
                encodings = face_recognition.face_encodings(rgb, locations_full)

                best_distance = 1.0
                best_loc = None
                for enc, loc in zip(encodings, locations_full):
                    dist = face_recognition.face_distance(
                        [self.registered_face_encoding], enc
                    )[0]
                    if dist < best_distance:
                        best_distance = dist
                        best_loc = loc

                # Convert distance → confidence (0–100 %)
                # distance 0 = perfect match, distance 0.6+ = no match
                confidence_pct = max(0.0, (1.0 - best_distance / 0.6)) * 100
                self.face_match_confidence = round(confidence_pct, 1)
                matched = best_distance < 0.50   # tunable threshold

                self.face_match_status = "MATCH" if matched else "NO_MATCH"

                # Draw bounding box with colour feedback
                if best_loc:
                    top, right, bottom, left = best_loc
                    box_color = (0, 220, 80) if matched else (0, 60, 255)
                    cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)
                    label = f"{'✓ MATCH' if matched else '✗ NO MATCH'}  {self.face_match_confidence:.0f}%"
                    cv2.putText(frame, label, (left, max(top - 10, 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, box_color, 2)

        # ── Status overlay at bottom ─────────────────────────────
        STATUS_COLORS = {
            "MATCH":            (0, 220, 80),
            "NO_MATCH":         (0, 60, 255),
            "SCANNING":         (0, 200, 255),
            "NO_FACE_DETECTED": (0, 165, 255),
        }
        color = STATUS_COLORS.get(self.face_match_status, (200, 200, 200))
        status_text = {
            "MATCH":            f"FACE VERIFIED — {self.face_match_confidence:.0f}% CONFIDENCE",
            "NO_MATCH":         f"FACE MISMATCH — {self.face_match_confidence:.0f}% (need >50%)",
            "SCANNING":         "SCANNING — ALIGN YOUR FACE WITH THE BOX",
            "NO_FACE_DETECTED": "NO FACE DETECTED — MOVE CLOSER",
        }.get(self.face_match_status, self.face_match_status)

        cv2.putText(frame, status_text, (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, color, 2)

        # ── Confidence bar ───────────────────────────────────────
        bar_w = int((w - 40) * self.face_match_confidence / 100)
        cv2.rectangle(frame, (20, h - 12), (w - 20, h - 6), (40, 40, 40), cv2.FILLED)
        if bar_w > 0:
            cv2.rectangle(frame, (20, h - 12), (20 + bar_w, h - 6), color, cv2.FILLED)

        return frame

    # ──────────────────────────────────────────────────────────────
    # Stage 2 — Passive Liveness (rPPG placeholder)
    # ──────────────────────────────────────────────────────────────
    def _stage_pulse(self, frame, rgb):
        results = self.face_mesh.process(rgb)
        if results.multi_face_landmarks:
            cv2.putText(frame, "ANALYZING BIOMETRIC PULSE...", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            self.hold_timer += 1
            cv2.rectangle(frame, (30, 70),
                          (30 + int((self.hold_timer / 60) * 300), 85),
                          (0, 255, 0), cv2.FILLED)
            if self.hold_timer > 60:
                self.state = 'GESTURE'
                self.hold_timer = 0
        else:
            cv2.putText(frame, "ALIGN YOUR FACE WITH THE CAMERA", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        return frame

    # ──────────────────────────────────────────────────────────────
    # Stage 3 — Gesture Challenge
    # ──────────────────────────────────────────────────────────────
    def _stage_gesture(self, frame, rgb):
        results = self.hands.process(rgb)
        cv2.putText(frame, f"CHALLENGE: SHOW {self.gesture_target} FINGERS", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
        cv2.putText(frame, f"Progress: {self.gesture_count}/{self.gesture_goal}", (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks, results.multi_handedness
            ):
                self.mp_draw.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )
                tip_ids = [4, 8, 12, 16, 20]
                fingers = []
                is_right = handedness.classification[0].label == 'Right'
                thumb_tip = hand_landmarks.landmark[tip_ids[0]]
                thumb_ip  = hand_landmarks.landmark[tip_ids[0] - 1]
                if is_right:
                    fingers.append(1 if thumb_tip.x < thumb_ip.x else 0)
                else:
                    fingers.append(1 if thumb_tip.x > thumb_ip.x else 0)

                for i in range(1, 5):
                    fingers.append(
                        1 if hand_landmarks.landmark[tip_ids[i]].y
                             < hand_landmarks.landmark[tip_ids[i] - 2].y
                        else 0
                    )

                current_fingers = sum(fingers)
                if current_fingers == self.gesture_target:
                    self.hold_timer += 1
                    cv2.rectangle(frame, (30, 110),
                                  (30 + self.hold_timer * 15, 130),
                                  (0, 255, 0), cv2.FILLED)
                    if self.hold_timer > 20:
                        self.gesture_count += 1
                        self.hold_timer = 0
                        if self.gesture_count >= self.gesture_goal:
                            self.state = 'HEAD_POSE'
                        else:
                            new_t = self.gesture_target
                            while new_t == self.gesture_target:
                                new_t = random.randint(1, 5)
                            self.gesture_target = new_t
                else:
                    self.hold_timer = 0
        return frame

    # ──────────────────────────────────────────────────────────────
    # Stage 4 — 3D Head-Pose Sequence
    # ──────────────────────────────────────────────────────────────
    def _stage_head(self, frame, rgb, w, h):
        results = self.face_mesh.process(rgb)
        direction = "CENTER"

        if results.multi_face_landmarks:
            for face_lm in results.multi_face_landmarks:
                face_2d, face_3d = [], []
                for idx in [1, 199, 33, 263, 61, 291]:
                    lm = face_lm.landmark[idx]
                    x, y = int(lm.x * w), int(lm.y * h)
                    face_2d.append([x, y])
                    face_3d.append([x, y, lm.z])

                face_2d = np.array(face_2d, dtype=np.float64)
                face_3d = np.array(face_3d, dtype=np.float64)
                cam_matrix = np.array(
                    [[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float64
                )
                success, rot_vec, _ = cv2.solvePnP(
                    face_3d, face_2d, cam_matrix,
                    np.zeros((4, 1), dtype=np.float64)
                )
                rmat, _ = cv2.Rodrigues(rot_vec)
                angles, *_ = cv2.RQDecomp3x3(rmat)
                pitch, yaw = angles[0] * 360, angles[1] * 360

                if yaw < -18:   direction = "LEFT"
                elif yaw > 18:  direction = "RIGHT"
                else:           direction = "CENTER"

                nx = int(face_lm.landmark[1].x * w)
                ny = int(face_lm.landmark[1].y * h)
                cv2.line(frame, (nx, ny),
                         (int(nx + yaw * 2), int(ny - pitch * 2)),
                         (255, 165, 0), 3)

        if self.strikes > 0:
            cv2.putText(frame, f"WARNING: {self.strikes}/2 STRIKES",
                        (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if self.awaiting_center:
            cv2.putText(frame, "RETURN TO CENTER", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
            if direction == "CENTER":
                self.hold_timer += 1
                if self.hold_timer >= 5:
                    self.awaiting_center = False
                    self.hold_timer = 0
            else:
                self.hold_timer = 0
        else:
            target = self.head_sequence[self.head_step]
            cv2.putText(frame,
                        f"STEP {self.head_step + 1}/{len(self.head_sequence)}: TURN {target}",
                        (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)

            if direction == target:
                self.hold_timer += 1
                cv2.rectangle(frame, (30, 80),
                              (30 + self.hold_timer * 20, 100),
                              (0, 255, 0), -1)
                if self.hold_timer >= 10:
                    self.head_step += 1
                    self.hold_timer = 0
                    if self.head_step >= len(self.head_sequence):
                        self.state = 'SUCCESS'
                    else:
                        self.awaiting_center = True
            elif direction in ("LEFT", "RIGHT") and direction != target:
                self.hold_timer += 1
                cv2.rectangle(frame, (30, 80),
                              (30 + self.hold_timer * 20, 100),
                              (0, 0, 255), -1)
                if self.hold_timer >= 10:
                    self.strikes += 1
                    self.hold_timer = 0
                    self.awaiting_center = True
                    if self.strikes >= 2:
                        self.state = 'FAILED'
            else:
                self.hold_timer = 0

        return frame

    # ──────────────────────────────────────────────────────────────
    # Terminal states
    # ──────────────────────────────────────────────────────────────
    def _stage_success(self, frame, w, h):
        cv2.rectangle(frame, (0, 0), (w, h), (0, 220, 80), 10)
        cv2.putText(frame, "IDENTITY VERIFIED",
                    (w // 2 - 160, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 220, 80), 4)
        return frame

    def _stage_failed(self, frame, w, h):
        cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 10)
        cv2.putText(frame, "KYC FAILED: ANOMALY DETECTED",
                    (w // 2 - 280, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 4)
        return frame

    # ──────────────────────────────────────────────────────────────
    # Reset — preserves the registered face encoding across sessions
    # ──────────────────────────────────────────────────────────────
    def reset(self):
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
        self._frame_count = 0

        # Reset match state but KEEP the registered encoding
        if self.registered_face_encoding is not None:
            self.face_match_status = "SCANNING"
        self.face_match_confidence = 0.0