from flask import Flask, render_template, Response, jsonify, request
import cv2
from kyc_engine import KYCEngine

app = Flask(__name__)
engine = KYCEngine()

@app.route('/')
def index():
    return render_template('index.html')

def gen():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success: break
        
        frame = cv2.flip(frame, 1)
        processed_frame = engine.process_frame(frame)
        
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_status')
def get_status():
    # Frontend polls this to update UI elements outside the video
    return jsonify({
        "state": engine.state,
        "strikes": engine.strikes,
        "progress": f"{engine.gesture_count}/{engine.gesture_goal}"
    })

@app.route('/next_stage', methods=['POST'])
def next_stage():
    # Logic to manually move past ID scan for prototype
    if engine.state == 'ID_SCAN':
        engine.state = 'PULSE'
    return jsonify({"success": True})

@app.route('/reset', methods=['POST'])
def reset_engine():
    engine.reset()
    return jsonify({"success": True})

if __name__ == '__main__':
    app.run(debug=True, port=5000)