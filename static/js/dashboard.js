/**
 * Dashboard live-camera scanning via the browser's getUserMedia API.
 * Sends frames to /api/scan and renders bounding boxes + emotion labels.
 */

(function () {
    "use strict";

    const video   = document.getElementById("webcam");
    const overlay = document.getElementById("overlay");
    const resultsDiv = document.getElementById("scan-results");
    const statusEl   = document.getElementById("scan-status");

    let stream = null;

    // ----------------------------------------------------------------
    //  Start the browser camera
    // ----------------------------------------------------------------
    async function initCamera() {
        try {
            stream = await navigator.mediaDevices.getUserMedia({
                video: { width: 640, height: 480 }
            });
            if (video) video.srcObject = stream;
        } catch (err) {
            console.warn("Camera not available:", err.message);
            if (statusEl) statusEl.textContent = "Camera not available";
        }
    }

    // ----------------------------------------------------------------
    //  Capture current frame as base64 JPEG
    // ----------------------------------------------------------------
    function grabFrame() {
        if (!video || video.readyState < 2) return null;
        const cvs = document.createElement("canvas");
        cvs.width  = video.videoWidth;
        cvs.height = video.videoHeight;
        cvs.getContext("2d").drawImage(video, 0, 0);
        return cvs.toDataURL("image/jpeg", 0.8);
    }

    // ----------------------------------------------------------------
    //  Draw bounding boxes on the overlay canvas
    // ----------------------------------------------------------------
    function drawBoxes(faces) {
        if (!overlay) return;
        const ctx = overlay.getContext("2d");
        overlay.width  = video.videoWidth;
        overlay.height = video.videoHeight;
        ctx.clearRect(0, 0, overlay.width, overlay.height);

        const scaleX = overlay.clientWidth  / video.videoWidth;
        const scaleY = overlay.clientHeight / video.videoHeight;

        faces.forEach(f => {
            const { top, right, bottom, left } = f.bbox;
            const x = left;
            const y = top;
            const w = right - left;
            const h = bottom - top;

            // Box
            ctx.strokeStyle = "#4361ee";
            ctx.lineWidth = 2;
            ctx.strokeRect(x, y, w, h);

            // Label background
            const label = `${f.emotion} (${Math.round(f.confidence * 100)}%)`;
            ctx.font = "bold 14px sans-serif";
            const tm = ctx.measureText(label);
            ctx.fillStyle = "rgba(67,97,238,.85)";
            ctx.fillRect(x, y - 20, tm.width + 10, 20);

            // Label text
            ctx.fillStyle = "#fff";
            ctx.fillText(label, x + 5, y - 5);
        });
    }

    // ----------------------------------------------------------------
    //  Render scan results below the camera
    // ----------------------------------------------------------------
    function renderResults(faces) {
        if (!resultsDiv) return;
        if (!faces.length) {
            resultsDiv.innerHTML = '<p class="empty-state">No face detected.</p>';
            return;
        }
        resultsDiv.innerHTML = faces.map(f => {
            const who = f.visitor_name
                ? `<strong>${f.visitor_name}</strong> (ID ${f.visitor_id})`
                : '<em>Unknown visitor</em>';
            const checkinBtn = f.visitor_id
                ? `<button class="btn btn-sm btn-success" onclick="quickCheckin(${f.visitor_id},'${f.emotion}',${f.confidence})">Check In</button>`
                : `<a class="btn btn-sm btn-primary" href="/register">Register</a>`;
            return `
                <div class="scan-result-item">
                    <div>${who} &mdash;
                        <span class="emotion-badge emotion-${f.emotion.toLowerCase()}">${f.emotion}</span>
                        ${Math.round(f.confidence * 100)}%
                    </div>
                    <div>${checkinBtn}</div>
                </div>`;
        }).join("");
    }

    // ----------------------------------------------------------------
    //  Exposed: scan the current frame
    // ----------------------------------------------------------------
    window.scanFrame = async function () {
        if (statusEl) statusEl.textContent = "Scanning...";
        const img = grabFrame();
        if (!img) {
            if (statusEl) statusEl.textContent = "Camera not ready";
            return;
        }
        try {
            const resp = await fetch("/api/scan", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ image: img })
            });
            const data = await resp.json();
            if (statusEl) statusEl.textContent = `Found ${data.faces.length} face(s)`;
            drawBoxes(data.faces);
            renderResults(data.faces);
        } catch (err) {
            if (statusEl) statusEl.textContent = "Scan failed: " + err.message;
        }
    };

    // ----------------------------------------------------------------
    //  Exposed: quick check-in recognised visitor
    // ----------------------------------------------------------------
    window.quickCheckin = async function (visitorId, emotion, confidence) {
        try {
            const resp = await fetch("/api/quick_checkin", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ visitor_id: visitorId, emotion, confidence })
            });
            const data = await resp.json();
            if (data.log_id) {
                alert("Visitor checked in (Log #" + data.log_id + ")");
                location.reload();
            }
        } catch (err) {
            alert("Check-in failed: " + err.message);
        }
    };

    // ----------------------------------------------------------------
    //  Auto-start
    // ----------------------------------------------------------------
    if (video) initCamera();
})();
