import cv2
import mediapipe as mp
import numpy as np
from scipy import signal
import collections
import time

class PulseDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.FOREHEAD_IDXS = [10, 67, 109, 10, 338, 297] 

    def get_roi_mean(self, frame, landmarks, indices):
        h, w, _ = frame.shape
        points = [[int(landmarks.landmark[idx].x * w), int(landmarks.landmark[idx].y * h)] for idx in indices]
        
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(points)], 255)
        mean_val = cv2.mean(frame, mask=mask)
        return mean_val[1] # Green Channel

class SignalProcessor:
    def __init__(self, buffer_size=150): # 150 frames is approx 5 seconds at 30fps
        self.buffer_size = buffer_size
        self.data_buffer = collections.deque(maxlen=buffer_size)
        self.times = collections.deque(maxlen=buffer_size)

    def add_data(self, value, timestamp):
        self.data_buffer.append(value)
        self.times.append(timestamp)

    def get_bpm(self):
        if len(self.data_buffer) < self.buffer_size:
            return None # Need more data to calculate
        
        # 1. Calculate the actual FPS of the webcam
        time_diff = self.times[-1] - self.times[0]
        if time_diff == 0: return None
        
        actual_fps = self.buffer_size / time_diff

        # 2. Detrend the signal (remove slow lighting changes)
        detrended = signal.detrend(self.data_buffer)

        # 3. Butterworth Bandpass Filter (0.75 Hz to 3.0 Hz -> 45 to 180 BPM)
        nyquist = 0.5 * actual_fps
        low = 0.75 / nyquist
        high = 3.0 / nyquist
        
        if low <= 0 or high >= 1:
            return None # FPS is too unstable

        b, a = signal.butter(3, [low, high], btype='band')
        filtered = signal.filtfilt(b, a, detrended)

        # 4. FFT to find dominant frequency (Heart Rate)
        fft_data = np.fft.rfft(filtered)
        freqs = np.fft.rfftfreq(self.buffer_size, 1.0 / actual_fps)
        
        peak_idx = np.argmax(np.abs(fft_data))
        dominant_freq = freqs[peak_idx]
        
        bpm = dominant_freq * 60
        return bpm if 45 <= bpm <= 180 else None

# --- Execution Loop ---
cap = cv2.VideoCapture(0)
detector = PulseDetector()
processor = SignalProcessor()

print("Please sit still under good lighting...")

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    results = detector.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.multi_face_landmarks:
        face_lms = results.multi_face_landmarks[0]
        
        # Extract signal
        green_val = detector.get_roi_mean(frame, face_lms, detector.FOREHEAD_IDXS)
        
        # Add to processor buffer
        processor.add_data(green_val, time.time())
        
        # Calculate BPM
        bpm = processor.get_bpm()
        
        # UI Feedback
        buffer_fill = int((len(processor.data_buffer) / processor.buffer_size) * 100)
        
        if bpm:
            cv2.putText(frame, f"BPM: {bpm:.1f}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            cv2.putText(frame, "LIVENESS DETECTED", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        else:
            cv2.putText(frame, f"Buffering... {buffer_fill}%", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)

    cv2.imshow('Pulse-Guard Engine', frame)
    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()