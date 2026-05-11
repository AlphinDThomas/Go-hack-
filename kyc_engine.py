import cv2
import mediapipe as mp
import numpy as np
import random
import collections
import face_recognition   # pip install face_recognition


class KYCEngine:
    FACE_MATCH_THRESHOLD  = 0.52   # dlib distance: lower = stricter (0.6 is default)
    FACE_HOLD_REQUIRED    = 30     # consecutive confident frames before auto-advance
    FACE_PROCESS_EVERY    = 3      # only run recognition every N frames (perf)

    def __init__(self):
        self.mp_face   = mp.solutions.face_mesh
        self.face_mesh = self.mp_face.FaceMesh(refine_landmarks=True)
        self.mp_hands  = mp.solutions.hands
        self.hands     = self.mp_hands.Hands(max_num_hands=1)
        self.mp_draw   = mp.solutions.drawing_utils

        # ── Face registry ──────────────────────────────────────────────
        self.reference_encoding = None   # numpy array, 128-d
        self.registered_name    = "Unknown"

        # ── Runtime state ──────────────────────────────────────────────
        self.state = 'FACE_MATCH'
        self._reset_internals()

    # ─────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────

    def set_reference(self, encoding: np.ndarray, name: str = "User"):
        """Load a registered face encoding into the engine."""
        self.reference_encoding = encoding
        self.registered_name    = name

    def reset(self):
        self.state = 'FACE_MATCH'
        self._reset_internals()

    def _reset_internals(self):
        self.face_hold_timer    = 0
        self.face_frame_counter = 0
        self.face_match_score   = 0.0   # 0–100 %
        self.face_locations     = []
        self.face_last_dist     = 1.0

        self.pulse_buffer       = collections.deque(maxlen=150)
        self.pulse_times        = collections.deque(maxlen=150)
        self.hold_timer         = 0

        self.gesture_target = random.randint(1, 5)
        self.gesture_count  = 0
        self.gesture_goal   = 5

        self.head_sequence    = [random.choice(["LEFT", "RIGHT"]) for _ in range(3)]
        self.head_step        = 0
        self.awaiting_center  = False

        self.strikes = 0

    # ─────────────────────────────────────────────────────────────────
    # Frame dispatcher
    # ─────────────────────────────────────────────────────────────────

    def process_frame(self, frame):
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        dispatch = {
            'FACE_MATCH': self._stage_face_match,
            'PULSE':      self._stage_pulse,
            'GESTURE':    self._stage_gesture,
            'HEAD_POSE':  self._stage_head,
            'SUCCESS':    self._stage_success,
            'FAILED':     self._stage_failed,
        }
        fn = dispatch.get(self.state)
        return fn(frame, rgb, w, h) if fn else frame

    # ─────────────────────────────────────────────────────────────────
    # STAGE 1 — Real face match
    # ─────────────────────────────────────────────────────────────────

    def _stage_face_match(self, frame, rgb, w, h):
        if self.reference_encoding is None:
            # No user loaded — tell the operator
            cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 200), 4)
            cv2.putText(frame, "NO USER LOADED", (30, h // 2 - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 80, 255), 3)
            cv2.putText(frame, "Register a user first in the dashboard",
                        (30, h // 2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
            return frame

        self.face_frame_counter += 1

        # Run recognition every N frames for performance
        if self.face_frame_counter % self.FACE_PROCESS_EVERY == 0:
            small   = cv2.resize(rgb, (0, 0), fx=0.5, fy=0.5)  # halve res → 4× faster
            locs    = face_recognition.face_locations(small, model="hog")
            # Scale locations back up
            self.face_locations = [(t*2, r*2, b*2, l*2) for (t, r, b, l) in locs]

            if self.face_locations:
                encs = face_recognition.face_encodings(rgb, self.face_locations)
                if encs:
                    dist = face_recognition.face_distance(
                        [self.reference_encoding], encs[0]
                    )[0]
                    self.face_last_dist  = dist
                    self.face_match_score = max(0.0, (1.0 - dist / self.FACE_MATCH_THRESHOLD) * 100)
                    self.face_match_score = min(self.face_match_score, 100.0)
                else:
                    self.face_match_score = 0.0
            else:
                self.face_match_score = 0.0

        matched = self.face_last_dist < self.FACE_MATCH_THRESHOLD

        # ── Draw face box ──────────────────────────────────────────
        for (top, right, bottom, left) in self.face_locations:
            color = (0, 220, 80) if matched else (0, 80, 220)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

            label = (f"{self.registered_name}  {self.face_match_score:.0f}%"
                     if matched else f"No match  {self.face_match_score:.0f}%")
            cv2.rectangle(frame, (left, bottom - 26), (right, bottom), color, cv2.FILLED)
            cv2.putText(frame, label, (left + 6, bottom - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # ── Hold timer & progress bar ──────────────────────────────
        if matched:
            self.face_hold_timer += 1
        else:
            self.face_hold_timer = max(0, self.face_hold_timer - 1)

        bar_w = int((self.face_hold_timer / self.FACE_HOLD_REQUIRED) * (w - 60))
        cv2.rectangle(frame, (30, h - 40), (w - 30, h - 20), (40, 40, 40), cv2.FILLED)
        cv2.rectangle(frame, (30, h - 40), (30 + bar_w, h - 20), (0, 220, 80), cv2.FILLED)

        header = (f"VERIFYING: {self.registered_name}" if self.face_locations
                  else "LOOKING FOR FACE...")
        cv2.putText(frame, header, (30, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
        cv2.putText(frame, "HOLD STILL — FACE COMPARISON IN PROGRESS",
                    (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)

        if self.face_hold_timer >= self.FACE_HOLD_REQUIRED:
            self.state = 'PULSE'
            self.hold_timer = 0

        return frame

    # ─────────────────────────────────────────────────────────────────
    # STAGE 2 — Pulse (rPPG)
    # ─────────────────────────────────────────────────────────────────

    def _stage_pulse(self, frame, rgb, w, h):
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
        return frame

    # ─────────────────────────────────────────────────────────────────
    # STAGE 3 — Gesture challenge
    # ─────────────────────────────────────────────────────────────────

    def _stage_gesture(self, frame, rgb, w, h):
        results = self.hands.process(rgb)
        cv2.putText(frame, f"CHALLENGE: SHOW {self.gesture_target} FINGERS",
                    (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
        cv2.putText(frame, f"Progress: {self.gesture_count}/{self.gesture_goal}",
                    (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        if results.multi_hand_landmarks:
            for hand_lm, handedness in zip(results.multi_hand_landmarks,
                                           results.multi_handedness):
                self.mp_draw.draw_landmarks(frame, hand_lm, self.mp_hands.HAND_CONNECTIONS)
                tip_ids = [4, 8, 12, 16, 20]
                fingers = []
                is_right = handedness.classification[0].label == 'Right'
                thumb_tip = hand_lm.landmark[tip_ids[0]]
                thumb_ip  = hand_lm.landmark[tip_ids[0] - 1]
                fingers.append(1 if (is_right and thumb_tip.x < thumb_ip.x)
                                 or (not is_right and thumb_tip.x > thumb_ip.x) else 0)
                for i in range(1, 5):
                    fingers.append(
                        1 if hand_lm.landmark[tip_ids[i]].y
                           < hand_lm.landmark[tip_ids[i] - 2].y else 0
                    )

                if fingers.count(1) == self.gesture_target:
                    self.hold_timer += 1
                    bar = min(self.hold_timer * 15, 300)
                    cv2.rectangle(frame, (30, 110), (30 + bar, 130), (0, 255, 0), cv2.FILLED)
                    if self.hold_timer > 20:
                        self.gesture_count += 1
                        self.hold_timer = 0
                        if self.gesture_count >= self.gesture_goal:
                            self.state = 'HEAD_POSE'
                        else:
                            t = self.gesture_target
                            while t == self.gesture_target:
                                t = random.randint(1, 5)
                            self.gesture_target = t
                else:
                    self.hold_timer = 0
        return frame

    # ─────────────────────────────────────────────────────────────────
    # STAGE 4 — 3D Head pose
    # ─────────────────────────────────────────────────────────────────

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
                cam_matrix = np.array([[w, 0, w/2], [0, w, h/2], [0, 0, 1]], dtype=np.float64)
                _, rot_vec, _ = cv2.solvePnP(face_3d, face_2d, cam_matrix,
                                              np.zeros((4, 1), dtype=np.float64))
                rmat, _ = cv2.Rodrigues(rot_vec)
                angles, *_ = cv2.RQDecomp3x3(rmat)
                pitch, yaw = angles[0] * 360, angles[1] * 360

                direction = ("LEFT" if yaw < -18 else "RIGHT" if yaw > 18 else "CENTER")

                nx = int(face_lm.landmark[1].x * w)
                ny = int(face_lm.landmark[1].y * h)
                cv2.line(frame, (nx, ny),
                         (int(nx + yaw * 2), int(ny - pitch * 2)), (255, 165, 0), 3)

        if self.strikes > 0:
            cv2.putText(frame, f"WARNING: {self.strikes}/2 STRIKES", (30, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

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
                cv2.rectangle(frame, (30, 80), (30 + self.hold_timer * 20, 100),
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
                cv2.rectangle(frame, (30, 80), (30 + self.hold_timer * 20, 100),
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

    # ─────────────────────────────────────────────────────────────────
    # Terminal stages
    # ─────────────────────────────────────────────────────────────────

    def _stage_success(self, frame, rgb, w, h):
        cv2.rectangle(frame, (0, 0), (w, h), (0, 255, 0), 10)
        cv2.putText(frame, f"IDENTITY VERIFIED — {self.registered_name}",
                    (int(w/2) - 200, int(h/2)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 4)
        return frame

    def _stage_failed(self, frame, rgb, w, h):
        cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 10)
        cv2.putText(frame, "KYC FAILED — ANOMALY DETECTED",
                    (int(w/2) - 250, int(h/2)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 4)
        return frame