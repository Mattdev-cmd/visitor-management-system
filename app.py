"""
Main Flask application — routes, API endpoints, and streaming.
"""

import base64
import hashlib
import os
import functools
import uuid

import cv2
import numpy as np
from flask import (
    Flask,
    Response,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)

import config
from camera import Camera
from database import init_db, get_db
from services import face_service, emotion_service, visitor_service

# ------------------------------------------------------------------ #
#  App factory
# ------------------------------------------------------------------ #

app = Flask(__name__)
app.secret_key = config.SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH

# Global camera instance (lazy)
_camera: Camera | None = None


def _get_camera() -> Camera:
    global _camera
    if _camera is None:
        _camera = Camera()
        _camera.open()
    return _camera


# ------------------------------------------------------------------ #
#  Template helpers
# ------------------------------------------------------------------ #

@app.context_processor
def inject_now():
    from datetime import datetime
    return {"now": datetime.now}


# ------------------------------------------------------------------ #
#  Auth
# ------------------------------------------------------------------ #

def login_required(f):
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("user_id"):
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated


def superadmin_required(f):
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("user_id"):
            return redirect(url_for("login"))
        if session.get("role") != "superadmin":
            flash("Access denied. Superadmin only.", "error")
            return redirect(url_for("index"))
        return f(*args, **kwargs)
    return decorated


@app.route("/login", methods=["GET", "POST"])
def login():
    if session.get("user_id"):
        return redirect(url_for("index"))
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        pw_hash = hashlib.sha256(password.encode()).hexdigest()
        with get_db() as conn:
            user = conn.execute(
                "SELECT id, username, role FROM users WHERE username = ? AND password_hash = ?",
                (username, pw_hash)
            ).fetchone()
        if user:
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            session["role"] = user["role"]
            return redirect(url_for("index"))
        flash("Invalid username or password.", "error")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ------------------------------------------------------------------ #
#  Public pre-registration (no login required)
# ------------------------------------------------------------------ #

@app.route("/pre-register", methods=["GET", "POST"])
def pre_register():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        purpose = request.form.get("purpose", "").strip()
        contact = request.form.get("contact", "").strip()

        if not name or not purpose:
            flash("Name and purpose are required.", "error")
            return redirect(url_for("pre_register"))

        # Save uploaded photo
        photo_path = None
        uploaded = request.files.get("photo")
        if uploaded and uploaded.filename:
            ext = uploaded.filename.rsplit(".", 1)[-1].lower()
            if ext in config.ALLOWED_EXTENSIONS:
                filename = f"prereg_{uuid.uuid4().hex[:12]}.{ext}"
                save_path = os.path.join(config.UPLOAD_FOLDER, filename)
                uploaded.save(save_path)
                photo_path = f"uploads/{filename}"

        with get_db() as conn:
            conn.execute(
                "INSERT INTO pre_registrations (name, purpose, contact, photo_path) VALUES (?, ?, ?, ?)",
                (name, purpose, contact, photo_path)
            )

        flash("Your pre-registration has been submitted! The admin will review it before your arrival.", "success")
        return redirect(url_for("pre_register"))

    return render_template("pre_register.html")


# ------------------------------------------------------------------ #
#  Admin: pending pre-registrations
# ------------------------------------------------------------------ #

@app.route("/pending")
@login_required
def pending_list():
    with get_db() as conn:
        pending = conn.execute(
            "SELECT * FROM pre_registrations WHERE status = 'pending' ORDER BY submitted_at DESC"
        ).fetchall()
    return render_template("pending.html", pending=pending)


@app.route("/pending/approve/<int:prereg_id>", methods=["POST"])
@login_required
def approve_prereg(prereg_id):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM pre_registrations WHERE id = ?", (prereg_id,)).fetchone()
        if not row:
            flash("Pre-registration not found.", "error")
            return redirect(url_for("pending_list"))

        # Load photo as cv2 frame for face enrollment
        photo_frame = None
        if row["photo_path"]:
            full_path = os.path.join(config.BASE_DIR, "static", row["photo_path"])
            if os.path.exists(full_path):
                photo_frame = cv2.imread(full_path)

        visitor_id = visitor_service.register_visitor(
            row["name"], row["purpose"], row["contact"], photo_frame
        )

        # Run emotion analysis on the photo
        emotion, confidence = "Neutral", 0.0
        if photo_frame is not None:
            faces = face_service.detect_faces(photo_frame)
            if faces:
                face_crop = face_service.crop_face(photo_frame, faces[0])
                emotion, confidence = emotion_service.predict_emotion(face_crop)

        visitor_service.log_visit(visitor_id, emotion, confidence, photo_frame)

        conn.execute(
            "UPDATE pre_registrations SET status = 'approved', reviewed_at = datetime('now','localtime') WHERE id = ?",
            (prereg_id,)
        )

    flash(f"Pre-registration approved. '{row['name']}' is now registered.", "success")
    return redirect(url_for("pending_list"))


@app.route("/pending/reject/<int:prereg_id>", methods=["POST"])
@login_required
def reject_prereg(prereg_id):
    with get_db() as conn:
        conn.execute(
            "UPDATE pre_registrations SET status = 'rejected', reviewed_at = datetime('now','localtime') WHERE id = ?",
            (prereg_id,)
        )
    flash("Pre-registration rejected.", "success")
    return redirect(url_for("pending_list"))


# ------------------------------------------------------------------ #
#  User management (superadmin only)
# ------------------------------------------------------------------ #

@app.route("/users")
@superadmin_required
def users_list():
    with get_db() as conn:
        users = conn.execute("SELECT id, username, role, created_at FROM users ORDER BY id").fetchall()
    return render_template("users.html", users=users)


@app.route("/users/add", methods=["POST"])
@superadmin_required
def add_user():
    username = request.form.get("username", "").strip()
    password = request.form.get("password", "")
    role = request.form.get("role", "admin")
    if not username or not password:
        flash("Username and password are required.", "error")
        return redirect(url_for("users_list"))
    if role not in ("admin", "superadmin"):
        role = "admin"
    pw_hash = hashlib.sha256(password.encode()).hexdigest()
    try:
        with get_db() as conn:
            conn.execute("INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                         (username, pw_hash, role))
        flash(f"User '{username}' created as {role}.", "success")
    except Exception:
        flash(f"Username '{username}' already exists.", "error")
    return redirect(url_for("users_list"))


@app.route("/users/delete/<int:user_id>", methods=["POST"])
@superadmin_required
def delete_user(user_id):
    if user_id == session.get("user_id"):
        flash("You cannot delete your own account.", "error")
        return redirect(url_for("users_list"))
    with get_db() as conn:
        conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
    flash("User deleted.", "success")
    return redirect(url_for("users_list"))


# ------------------------------------------------------------------ #
#  Page routes
# ------------------------------------------------------------------ #

@app.route("/")
@login_required
def index():
    stats = visitor_service.get_dashboard_stats()
    recent_logs = visitor_service.get_visit_logs(limit=10)
    return render_template("dashboard.html", stats=stats, logs=recent_logs)

    # ------------------------------------------------------------------ #
    #  Visitor Logs Report
    # ------------------------------------------------------------------ #


@app.route("/logs")
@login_required
def logs_report():
    logs = visitor_service.get_visit_logs(limit=1000)
    return render_template("logs_report.html", logs=logs)

@app.route("/logs/export/pdf")
@login_required
def export_logs_pdf():
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    from io import BytesIO
    logs = visitor_service.get_visit_logs(limit=1000)
    buffer = BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    y = height - 40
    p.setFont("Helvetica-Bold", 14)
    p.drawString(40, y, "Visitor Logs Report")
    y -= 30
    p.setFont("Helvetica", 10)
    headers = ["Visitor", "Emotion", "Confidence", "Check-In", "Check-Out"]
    for i, h in enumerate(headers):
        p.drawString(40 + i*100, y, h)
    y -= 20
    for log in logs:
        if y < 50:
            p.showPage()
            y = height - 40
        p.drawString(40, y, str(log.get("visitor_name", "")))
        p.drawString(140, y, str(log.get("emotion", "")))
        p.drawString(240, y, f"{int(log.get('confidence',0)*100)}%")
        p.drawString(340, y, str(log.get("checked_in_at", "")))
        p.drawString(440, y, str(log.get("checked_out_at", "") or '—'))
        y -= 18
    p.save()
    buffer.seek(0)
    return Response(buffer, mimetype='application/pdf', headers={
        'Content-Disposition': 'attachment;filename=visitor_logs.pdf'
    })

@app.route("/logs/export/excel")
@login_required
def export_logs_excel():
    import pandas as pd
    from io import BytesIO
    logs = visitor_service.get_visit_logs(limit=1000)
    df = pd.DataFrame(logs)
    # Only keep relevant columns and rename
    df = df[["visitor_name", "emotion", "confidence", "checked_in_at", "checked_out_at"]]
    df.columns = ["Visitor", "Emotion", "Confidence", "Check-In", "Check-Out"]
    df["Confidence"] = (df["Confidence"] * 100).astype(int).astype(str) + "%"
    output = BytesIO()
    df.to_excel(output, index=False, engine="openpyxl")
    output.seek(0)
    return Response(output, mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', headers={
        'Content-Disposition': 'attachment;filename=visitor_logs.xlsx'
    })


@app.route("/register", methods=["GET", "POST"])
@login_required
def register():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        purpose = request.form.get("purpose", "").strip()
        contact = request.form.get("contact", "").strip()

        if not name or not purpose:
            flash("Name and purpose are required.", "error")
            return redirect(url_for("register"))

        # --- Photo: accept file upload OR base64 from webcam snapshot ---
        photo_frame = None

        uploaded = request.files.get("photo")
        if uploaded and uploaded.filename:
            data = uploaded.read()
            arr = np.frombuffer(data, np.uint8)
            photo_frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        b64 = request.form.get("photo_b64", "")
        if b64 and photo_frame is None:
            raw = base64.b64decode(b64.split(",")[-1])
            arr = np.frombuffer(raw, np.uint8)
            photo_frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        visitor_id = visitor_service.register_visitor(name, purpose, contact, photo_frame)

        # --- Emotion analysis on the registration photo ---
        emotion, confidence = "Neutral", 0.0
        if photo_frame is not None:
            faces = face_service.detect_faces(photo_frame)
            if faces:
                face_crop = face_service.crop_face(photo_frame, faces[0])
                emotion, confidence = emotion_service.predict_emotion(face_crop)

        visitor_service.log_visit(visitor_id, emotion, confidence, photo_frame)

        flash(f"Visitor '{name}' registered successfully! Detected emotion: {emotion}", "success")
        return redirect(url_for("index"))

    return render_template("register.html")


@app.route("/visitors")
@login_required
def visitors_list():
    q = request.args.get("q", "").strip()
    if q:
        visitors = visitor_service.search_visitors(q)
    else:
        visitors = visitor_service.get_all_visitors()
    return render_template("visitors.html", visitors=visitors, query=q)


@app.route("/visitor/<int:visitor_id>")
@login_required
def visitor_detail(visitor_id):
    visitor = visitor_service.get_visitor(visitor_id)
    if not visitor:
        flash("Visitor not found.", "error")
        return redirect(url_for("visitors_list"))
    logs = visitor_service.get_visit_logs(visitor_id=visitor_id)
    return render_template("visitor_detail.html", visitor=visitor, logs=logs)


@app.route("/checkout/<int:log_id>", methods=["POST"])
@login_required
def checkout(log_id):
    visitor_service.checkout_visitor(log_id)
    flash("Visitor checked out.", "success")
    return redirect(url_for("index"))


# ------------------------------------------------------------------ #
#  API / AJAX endpoints
# ------------------------------------------------------------------ #

@app.route("/api/scan", methods=["POST"])
def api_scan():
    """Accept a base64 image, detect face & emotion, try to identify."""
    data = request.get_json(force=True)
    b64 = data.get("image", "")
    if not b64:
        return jsonify({"error": "No image provided"}), 400

    raw = base64.b64decode(b64.split(",")[-1])
    arr = np.frombuffer(raw, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "Invalid image"}), 400

    faces = face_service.detect_faces(frame)
    if not faces:
        return jsonify({"faces": [], "message": "No face detected"})

    results = []
    for loc in faces:
        top, right, bottom, left = loc
        face_crop = face_service.crop_face(frame, loc)
        emotion, confidence = emotion_service.predict_emotion(face_crop)
        visitor_id = face_service.identify_visitor(frame, loc)
        visitor_name = None
        if visitor_id:
            v = visitor_service.get_visitor(visitor_id)
            visitor_name = v["name"] if v else None

        results.append({
            "bbox": {"top": top, "right": right, "bottom": bottom, "left": left},
            "emotion": emotion,
            "confidence": round(confidence, 2),
            "visitor_id": visitor_id,
            "visitor_name": visitor_name,
        })

    return jsonify({"faces": results})


@app.route("/api/quick_checkin", methods=["POST"])
def api_quick_checkin():
    """Auto-checkin a recognised visitor from a camera scan."""
    data = request.get_json(force=True)
    visitor_id = data.get("visitor_id")
    emotion = data.get("emotion", "Neutral")
    confidence = data.get("confidence", 0.0)
    if not visitor_id:
        return jsonify({"error": "visitor_id required"}), 400

    log_id = visitor_service.log_visit(visitor_id, emotion, confidence)
    return jsonify({"log_id": log_id, "status": "checked_in"})


@app.route("/api/stats")
def api_stats():
    return jsonify(visitor_service.get_dashboard_stats())


# ------------------------------------------------------------------ #
#  Camera stream
# ------------------------------------------------------------------ #

@app.route("/video_feed")
def video_feed():
    def gen(camera):
        while True:
            frame = camera.get_frame()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    try:
        cam = Camera()
        return Response(gen(cam), mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as exc:
        return str(exc), 500


# ------------------------------------------------------------------ #
#  Entry point
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    os.makedirs(config.UPLOAD_FOLDER, exist_ok=True)
    init_db()
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG, threaded=True)
