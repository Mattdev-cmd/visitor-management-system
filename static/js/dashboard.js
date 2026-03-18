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
    let autoDetectInterval = null;
    let isAutoDetecting = false;

    // ----------------------------------------------------------------
    //  Start the browser camera
    // ----------------------------------------------------------------
    async function initCamera() {
        try {
            stream = await navigator.mediaDevices.getUserMedia({
                video: { width: 640, height: 480 }
            });
            if (video) video.srcObject = stream;
            // Start auto-detection after camera initializes
            setTimeout(startAutoDetection, 1000);
        } catch (err) {
            console.warn("Camera not available:", err.message);
            if (statusEl) statusEl.textContent = "Camera not available";
        }
    }

    // ----------------------------------------------------------------
    //  Auto-detection: scan frames automatically every 800ms
    // ----------------------------------------------------------------
    function startAutoDetection() {
        if (isAutoDetecting) return;
        isAutoDetecting = true;
        if (statusEl) statusEl.textContent = "Auto-detecting...";
        autoDetectInterval = setInterval(async () => {
            const img = grabFrame();
            if (!img) return;
            try {
                const resp = await fetch("/api/scan", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ image: img })
                });
                const data = await resp.json();
                if (data.faces && data.faces.length > 0) {
                    if (statusEl) statusEl.textContent = `Detected ${data.faces.length} face(s)`;
                    drawBoxes(data.faces);
                    renderResults(data.faces);
                } else {
                    if (statusEl) statusEl.textContent = "No faces detected";
                    resultsDiv.innerHTML = '<p class="empty-state">No face detected.</p>';
                    // Clear boxes
                    if (overlay) {
                        const ctx = overlay.getContext("2d");
                        overlay.width = video.videoWidth;
                        overlay.height = video.videoHeight;
                        ctx.clearRect(0, 0, overlay.width, overlay.height);
                    }
                }
            } catch (err) {
                console.error("Auto-detect error:", err);
            }
        }, 800);
    }

    function stopAutoDetection() {
        if (autoDetectInterval) {
            clearInterval(autoDetectInterval);
            autoDetectInterval = null;
            isAutoDetecting = false;
            if (statusEl) statusEl.textContent = "Auto-detection stopped";
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

            // Color based on emotion
            let boxColor = "#4361ee"; // Default blue
            if (f.emotion === "Angry") {
                boxColor = "#ff4757"; // Red for angry
            } else if (f.emotion === "Happy") {
                boxColor = "#2ed573"; // Green for happy
            } else if (f.emotion === "Sad") {
                boxColor = "#a4d0e1"; // Light blue for sad
            }

            // Box
            ctx.strokeStyle = boxColor;
            ctx.lineWidth = 3;
            ctx.strokeRect(x, y, w, h);

            // Label background
            const label = `${f.emotion} (${Math.round(f.confidence * 100)}%)`;
            ctx.font = "bold 14px sans-serif";
            const tm = ctx.measureText(label);
            ctx.fillStyle = boxColor.replace(")", ", 0.85)").replace("rgb", "rgba");
            ctx.fillRect(x, y - 25, tm.width + 10, 25);

            // Label text
            ctx.fillStyle = "#fff";
            ctx.fillText(label, x + 5, y - 8);
        });
    }

    // ----------------------------------------------------------------
    //  Enhanced notification for Angry emotion
    // ----------------------------------------------------------------
    function showAngryNotification() {
        // Show alert banner
        let notif = document.getElementById("angry-alert");
        if (!notif) {
            notif = document.createElement("div");
            notif.id = "angry-alert";
            notif.className = "alert-emotion-angry";
            notif.innerHTML = `
                <div class="alert-content">
                    <strong>⚠️ ALERT:</strong> Angry emotion detected!
                </div>
            `;
            document.body.appendChild(notif);
        }
        notif.style.display = "block";
        notif.classList.add("pulse-animation");
        
        // Play alert sound if available
        try {
            const audio = new Audio("data:audio/wav;base64,UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAAB9AAACABAAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj==");
            audio.play().catch(() => {});
        } catch (e) {}
        
        setTimeout(() => { 
            notif.style.display = "none";
            notif.classList.remove("pulse-animation");
        }, 5000);
    }

    //  Render scan results below the camera
    // ----------------------------------------------------------------
    function renderResults(faces) {
        if (!resultsDiv) return;
        if (!faces.length) {
            resultsDiv.innerHTML = '<p class="empty-state">No face detected.</p>';
            return;
        }
        // Check for Angry emotion
        if (faces.some(f => f.emotion === "Angry")) {
            showAngryNotification();
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
    //  Exposed: scan the current frame (manual scan or internal use)
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
    //  Toggle auto-detection on/off
    // ----------------------------------------------------------------
    window.toggleAutoDetection = function() {
        const btn = document.getElementById("btn-scan");
        if (isAutoDetecting) {
            stopAutoDetection();
            btn.textContent = '▶ Start Auto-Detection';
            btn.classList.remove("btn-active");
        } else {
            startAutoDetection();
            btn.textContent = '⏹ Stop Auto-Detection';
            btn.classList.add("btn-active");
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
