from flask import Flask, render_template, Response, jsonify, request
import cv2
from kyc_engine import KYCEngine

app = Flask(__name__)
engine = KYCEngine()


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────
def gen():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.flip(frame, 1)
        processed = engine.process_frame(frame)
        ret, buffer = cv2.imencode('.jpg', processed)
        if not ret:
            continue
        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n'
            + buffer.tobytes()
            + b'\r\n'
        )


# ──────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/get_status')
def get_status():
    return jsonify({
        "state":                engine.state,
        "strikes":              engine.strikes,
        "progress":             f"{engine.gesture_count}/{engine.gesture_goal}",
        "gesture_target":       engine.gesture_target,
        "face_match_status":    engine.face_match_status,
        "face_match_confidence": engine.face_match_confidence,
        "user_registered":      engine.registered_face_encoding is not None,
    })


@app.route('/register_face', methods=['POST'])
def register_face():
    """
    Accepts a multipart/form-data upload with field name 'photo'.
    Extracts the face encoding and stores it in the engine.
    Returns JSON: { success: bool, message: str }
    """
    if 'photo' not in request.files:
        return jsonify({"success": False, "message": "No file uploaded (field name: 'photo')."}), 400

    file = request.files['photo']
    if file.filename == '':
        return jsonify({"success": False, "message": "Empty filename."}), 400

    allowed = {'jpg', 'jpeg', 'png', 'webp'}
    ext = file.filename.rsplit('.', 1)[-1].lower()
    if ext not in allowed:
        return jsonify({"success": False,
                        "message": f"Unsupported format '{ext}'. Use JPG, PNG, or WebP."}), 400

    image_bytes = file.read()
    success, message = engine.register_face(image_bytes)
    status_code = 200 if success else 422
    return jsonify({"success": success, "message": message}), status_code


@app.route('/next_stage', methods=['POST'])
def next_stage():
    """
    Advances from ID_SCAN → PULSE only when the live face actually matches
    the registered encoding.
    """
    if engine.state != 'ID_SCAN':
        return jsonify({"success": True})   # already past this stage

    # Guard: no user enrolled
    if engine.registered_face_encoding is None:
        return jsonify({
            "success": False,
            "error": "no_user_registered",
            "message": "No user has been registered. Upload a photo first."
        }), 400

    # Guard: face doesn't match
    if engine.face_match_status != "MATCH":
        status_labels = {
            "NO_MATCH":         f"Face does not match the registered user ({engine.face_match_confidence:.0f}% confidence — need >50%).",
            "NO_FACE_DETECTED": "No face is visible in the camera feed.",
            "SCANNING":         "Scanner is still initialising. Wait a moment and try again.",
        }
        msg = status_labels.get(engine.face_match_status,
                                "Identity could not be confirmed. Try again.")
        return jsonify({
            "success": False,
            "error": "face_mismatch",
            "message": msg
        }), 403

    # All good — advance
    engine.state = 'PULSE'
    return jsonify({"success": True})


@app.route('/reset', methods=['POST'])
def reset_engine():
    engine.reset()
    return jsonify({"success": True})


# ──────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True, port=5000)