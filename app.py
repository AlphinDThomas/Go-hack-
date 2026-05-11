from dotenv import load_dotenv
load_dotenv()

import os
import uuid
import base64
import json
import numpy as np
import cv2
import face_recognition
from flask import Flask, render_template, Response, jsonify, request
from flask_cors import CORS
from supabase import create_client, Client

from kyc_engine import KYCEngine

app    = Flask(__name__)
CORS(app)
engine = KYCEngine()

# ── Supabase client ───────────────────────────────────────────────────────────
# Set these in Railway environment variables (never hardcode)
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def db_get_all_users():
    res = supabase.table("users").select("user_id, name").execute()
    return res.data or []

def db_get_user(user_id: str):
    res = supabase.table("users").select("*").eq("user_id", user_id).execute()
    return res.data[0] if res.data else None

def db_insert_user(user_id: str, name: str, encoding: list):
    supabase.table("users").insert({
        "user_id":  user_id,
        "name":     name,
        "encoding": encoding          # jsonb column — list of 128 floats
    }).execute()

def db_delete_user(user_id: str):
    supabase.table("users").delete().eq("user_id", user_id).execute()

# ─────────────────────────────────────────────────────────────────────────────
# Pages
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')

# ─────────────────────────────────────────────────────────────────────────────
# NEW: Frame-based processing endpoint
# Browser captures webcam frame → sends as base64 JPEG → server processes →
# returns annotated frame as base64 + status JSON.
# Replaces the old MJPEG /video_feed stream (incompatible with Railway/cloud).
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/process_frame', methods=['POST'])
def process_frame():
    data = request.get_json(silent=True) or {}
    frame_b64 = data.get('frame', '')

    if not frame_b64:
        return jsonify({"error": "No frame provided"}), 400

    # Decode base64 → numpy BGR image
    try:
        img_bytes = base64.b64decode(frame_b64.split(',')[-1])  # strip data:image/jpeg;base64,
        img_arr   = np.frombuffer(img_bytes, np.uint8)
        frame     = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("decode failed")
    except Exception:
        return jsonify({"error": "Invalid frame data"}), 400

    # Run KYC engine on the frame
    processed = engine.process_frame(frame)

    # Encode processed frame back to base64 JPEG
    _, buf     = cv2.imencode('.jpg', processed, [cv2.IMWRITE_JPEG_QUALITY, 75])
    out_b64    = 'data:image/jpeg;base64,' + base64.b64encode(buf).decode('utf-8')

    return jsonify({
        "frame":         out_b64,
        "state":         engine.state,
        "strikes":       engine.strikes,
        "progress":      f"{engine.gesture_count}/{engine.gesture_goal}",
        "gesture_count": engine.gesture_count,
        "gesture_goal":  engine.gesture_goal,
        "head_step":     engine.head_step,
        "head_total":    len(engine.head_sequence),
        "gesture_target":engine.gesture_target,
        "face_score":    round(engine.face_match_score, 1),
        "loaded_user":   engine.registered_name,
    })

# ─────────────────────────────────────────────────────────────────────────────
# Register user  POST /register
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/register', methods=['POST'])
def register_user():
    name = request.form.get('name', '').strip()
    if not name:
        return jsonify({"error": "Name is required"}), 400

    photo_file = request.files.get('photo')
    if not photo_file:
        return jsonify({"error": "Photo is required"}), 400

    file_bytes = np.frombuffer(photo_file.read(), np.uint8)
    img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return jsonify({"error": "Could not decode image"}), 400

    img_rgb   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    locations = face_recognition.face_locations(img_rgb, model="hog")

    if not locations:
        return jsonify({"error": "No face detected. Use a clear, well-lit frontal photo."}), 422
    if len(locations) > 1:
        return jsonify({"error": "Multiple faces detected. Upload a photo with one person only."}), 422

    encoding = face_recognition.face_encodings(img_rgb, locations)[0]
    user_id  = str(uuid.uuid4())[:8].upper()

    db_insert_user(user_id, name, encoding.tolist())

    return jsonify({"user_id": user_id, "name": name, "success": True})

# ─────────────────────────────────────────────────────────────────────────────
# List users  GET /users
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/users')
def list_users():
    return jsonify(db_get_all_users())

# ─────────────────────────────────────────────────────────────────────────────
# Delete user  DELETE /users/<user_id>
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/users/<user_id>', methods=['DELETE'])
def delete_user(user_id):
    if not db_get_user(user_id):
        return jsonify({"error": "User not found"}), 404
    db_delete_user(user_id)
    return jsonify({"success": True})

# ─────────────────────────────────────────────────────────────────────────────
# Load user into engine  POST /load_user
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/load_user', methods=['POST'])
def load_user():
    data    = request.get_json(silent=True) or {}
    user_id = data.get('user_id', '').strip().upper()

    user = db_get_user(user_id)
    if not user:
        return jsonify({"error": "User not found"}), 404

    encoding = np.array(user['encoding'], dtype=np.float64)
    engine.set_reference(encoding, name=user['name'])
    engine.reset()

    return jsonify({"success": True, "name": user['name']})

# ─────────────────────────────────────────────────────────────────────────────
# Status poll  GET /get_status  (still available as fallback)
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/get_status')
def get_status():
    return jsonify({
        "state":          engine.state,
        "strikes":        engine.strikes,
        "progress":       f"{engine.gesture_count}/{engine.gesture_goal}",
        "gesture_count":  engine.gesture_count,
        "gesture_goal":   engine.gesture_goal,
        "head_step":      engine.head_step,
        "head_total":     len(engine.head_sequence),
        "gesture_target": engine.gesture_target,
        "face_score":     round(engine.face_match_score, 1),
        "loaded_user":    engine.registered_name,
    })

# ─────────────────────────────────────────────────────────────────────────────
# Reset  POST /reset
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/reset', methods=['POST'])
def reset_engine():
    engine.reset()
    return jsonify({"success": True})

# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)