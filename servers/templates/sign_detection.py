from .base import render_template


_EXTRA_CSS = """
.video-wrapper {
    position: relative;
    width: 100%;
}

.stop-banner {
    position: absolute;
    top: 12px;
    left: 12px;
    right: 12px;
    z-index: 5;
    display: none;
    padding: 10px 14px;
    border-radius: 6px;
    background: rgba(231, 76, 60, 0.92);
    color: white;
    font-size: 14px;
    font-weight: 700;
    text-align: center;
}

.stop-banner.active {
    display: block;
}

.stream.stopped {
    outline: 4px solid #e74c3c;
}

.key-display {
    display: grid;
    grid-template-columns: repeat(3, 42px);
    grid-template-rows: repeat(2, 42px);
    gap: 6px;
    justify-content: center;
    margin-top: 10px;
}

.key-box {
    display: flex;
    align-items: center;
    justify-content: center;
    background: #222;
    border: 1px solid #444;
    border-radius: 5px;
    color: #888;
    font-size: 18px;
    font-weight: 700;
}

.key-box.active {
    background: #2ecc71;
    color: #111;
}

.key-up {
    grid-column: 2;
    grid-row: 1;
}

.key-left {
    grid-column: 1;
    grid-row: 2;
}

.key-down {
    grid-column: 2;
    grid-row: 2;
}

.key-right {
    grid-column: 3;
    grid-row: 2;
}

.debug-panel {
    font-family: "Courier New", monospace;
    font-size: 11px;
    background: var(--bg-sidebar, #111);
    border: 1px solid var(--border-color, #333);
    border-radius: 5px;
    padding: 8px;
    max-height: 260px;
    overflow-y: auto;
}

.debug-line {
    display: flex;
    justify-content: space-between;
    gap: 10px;
    padding: 2px 0;
    border-bottom: 1px solid rgba(255,255,255,0.04);
}

.debug-line:last-child {
    border-bottom: 0;
}

.debug-key {
    color: #58a6ff;
    white-space: nowrap;
}

.debug-value {
    color: var(--text-secondary, #ddd);
    text-align: right;
    word-break: break-all;
}

.log-panel {
    font-family: "Courier New", monospace;
    font-size: 11px;
    background: var(--bg-sidebar, #111);
    border: 1px solid var(--border-color, #333);
    border-radius: 5px;
    padding: 8px;
    max-height: 240px;
    overflow-y: auto;
    display: flex;
    flex-direction: column-reverse;
}

.log-line {
    padding: 2px 0;
    color: var(--text-muted, #888);
    white-space: pre-wrap;
    overflow-wrap: anywhere;
}

.log-ts {
    color: #666;
    margin-right: 6px;
}

.log-ok {
    color: #2ecc71;
}

.log-warn {
    color: #f1c40f;
}

.log-error {
    color: #e74c3c;
}

.log-info {
    color: #58a6ff;
}

.log-state {
    color: #c77dff;
}

.det-row {
    display: grid;
    grid-template-columns: 80px 50px 1fr;
    gap: 8px;
    padding: 6px 0;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    font-size: 12px;
}

.det-class {
    font-weight: 700;
    color: #58a6ff;
}

.det-score {
    color: #2ecc71;
}

.det-bbox {
    color: var(--text-muted, #888);
    font-family: monospace;
}

.status-chip {
    display: inline-block;
    padding: 3px 7px;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 700;
    background: #333;
    color: #aaa;
}

.status-chip.ok {
    background: rgba(46, 204, 113, 0.18);
    color: #2ecc71;
}

.status-chip.bad {
    background: rgba(231, 76, 60, 0.18);
    color: #e74c3c;
}

.status-chip.warn {
    background: rgba(241, 196, 15, 0.18);
    color: #f1c40f;
}
"""


CONTENT = """
<div class="container">
    <div class="video-section">
        <div class="video-wrapper">
            <div id="stop-banner" class="stop-banner"></div>
            <img src="{{ url_for('video') }}" id="stream-img" class="stream" alt="Sign Detection Stream">
        </div>
    </div>

    <div class="controls-section">

        <div class="card">
            <div class="card-header">Drive Control</div>

            <div style="display:flex;align-items:center;gap:12px;margin-bottom:12px">
                <span id="run-indicator" style="display:inline-block;width:14px;height:14px;border-radius:50%;background:#e74c3c"></span>
                <span id="run-label" style="font-size:14px;font-weight:700;color:var(--text-secondary)">STOPPED</span>
            </div>

            <div style="display:flex;gap:10px;margin-bottom:8px">
                <button onclick="driveStart()" class="button success" style="flex:1">Start</button>
                <button onclick="driveStop()" class="button" style="flex:1;background:var(--accent-orange,#e67e22)">Stop</button>
            </div>

            <button id="mode-btn" onclick="toggleMode()" class="button" style="width:100%;background:#555">Manual</button>

            <div id="key-panel" style="display:none">
                <div class="key-display">
                    <div class="key-box key-up" id="key-up">&#9650;</div>
                    <div class="key-box key-left" id="key-left">&#9664;</div>
                    <div class="key-box key-down" id="key-down">&#9660;</div>
                    <div class="key-box key-right" id="key-right">&#9654;</div>
                </div>
                <p style="text-align:center;font-size:11px;color:var(--text-muted);margin:6px 0 0">
                    Arrow keys or WASD
                </p>
            </div>
        </div>

        <div class="card">
            <div class="card-header">Runtime Status</div>
            <div id="runtime-debug" class="debug-panel">
                <div class="debug-line">
                    <span class="debug-key">status</span>
                    <span class="debug-value">Waiting...</span>
                </div>
            </div>
        </div>

        <div class="card">
            <div class="card-header">
                Lane Debug
                <span id="lane-state-chip" class="status-chip">UNKNOWN</span>
            </div>
            <div id="lane-debug" class="debug-panel">
                <div class="debug-line">
                    <span class="debug-key">lane</span>
                    <span class="debug-value">Waiting...</span>
                </div>
            </div>
        </div>

        <div class="card">
            <div class="card-header">
                Sign FSM Debug
                <span id="sign-state-chip" class="status-chip">UNKNOWN</span>
            </div>
            <div id="sign-debug" class="debug-panel">
                <div class="debug-line">
                    <span class="debug-key">sign</span>
                    <span class="debug-value">Waiting...</span>
                </div>
            </div>
        </div>

        <div class="card">
            <div class="card-header">
                Detections
                <span id="det-count" style="font-size:11px;font-weight:400;color:var(--text-muted)"></span>
            </div>
            <div id="detections" class="detections-list">
                <div class="empty-state">Waiting for detections...</div>
            </div>
        </div>

        <div class="card">
            <div class="card-header">
                Bot Logs
                <button onclick="clearLogs()" style="font-size:10px;padding:2px 8px;background:#222;border:1px solid #444;color:#888;border-radius:3px;cursor:pointer;margin-left:auto">
                    Clear
                </button>
            </div>
            <div id="bot-logs" class="log-panel">
                <div class="log-line">Waiting for status...</div>
            </div>
        </div>

    </div>
</div>
"""


EXTRA_JS = """
function postJSON(url, data) {
    return fetch(url, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(data || {})
    }).then(r => r.json());
}

function esc(s) {
    return String(s)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');
}

function fmt(v) {
    if (v === null || v === undefined) return 'null';
    if (typeof v === 'number') return Number.isInteger(v) ? String(v) : v.toFixed(3);
    if (typeof v === 'boolean') return v ? 'true' : 'false';
    if (Array.isArray(v)) return '[' + v.join(', ') + ']';
    if (typeof v === 'object') return JSON.stringify(v);
    return String(v);
}

function renderDebugPanel(id, rows) {
    const el = document.getElementById(id);
    if (!el) return;

    el.innerHTML = rows.map(([key, value]) => {
        return '<div class="debug-line">' +
            '<span class="debug-key">' + esc(key) + '</span>' +
            '<span class="debug-value">' + esc(fmt(value)) + '</span>' +
            '</div>';
    }).join('');
}

function setChip(id, text, mode) {
    const el = document.getElementById(id);
    if (!el) return;

    el.textContent = text || 'UNKNOWN';
    el.className = 'status-chip ' + (mode || '');
}

function setRunningUI(isRunning) {
    const indicator = document.getElementById('run-indicator');
    const label = document.getElementById('run-label');

    indicator.style.background = isRunning ? '#2ecc71' : '#e74c3c';
    label.textContent = isRunning ? 'RUNNING' : 'STOPPED';
    label.style.color = isRunning ? '#2ecc71' : 'var(--text-secondary)';
}

function driveStart() {
    postJSON('/start', {}).then(() => {
        setRunningUI(true);
        addLog('[Server] start pressed', 'ok');
    }).catch(() => {
        addLog('[Server] start failed', 'error');
    });
}

function driveStop() {
    postJSON('/stop', {}).then(() => {
        setRunningUI(false);
        addLog('[Server] stop pressed', 'warn');
    }).catch(() => {
        addLog('[Server] stop failed', 'error');
    });
}

let manualMode = false;

function toggleMode() {
    manualMode = !manualMode;

    postJSON('/set_mode', {mode: manualMode ? 'manual' : 'auto'}).then(data => {
        manualMode = data.mode === 'manual';

        const btn = document.getElementById('mode-btn');
        const panel = document.getElementById('key-panel');

        btn.textContent = manualMode ? 'Auto' : 'Manual';
        panel.style.display = manualMode ? 'block' : 'none';

        addLog('[Server] mode=' + data.mode, 'info');
    });
}

const keyState = {
    up: false,
    down: false,
    left: false,
    right: false
};

const keyMap = {
    'ArrowUp': 'up',
    'ArrowDown': 'down',
    'ArrowLeft': 'left',
    'ArrowRight': 'right',
    'w': 'up',
    'W': 'up',
    's': 'down',
    'S': 'down',
    'a': 'left',
    'A': 'left',
    'd': 'right',
    'D': 'right'
};

function updateKeyDisplay() {
    for (const [key, active] of Object.entries(keyState)) {
        const el = document.getElementById('key-' + key);
        if (el) el.classList.toggle('active', active);
    }
}

function sendKeys() {
    fetch('/keys', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(keyState)
    }).catch(() => {});
}

document.addEventListener('keydown', e => {
    const dir = keyMap[e.key];
    if (!dir) return;

    e.preventDefault();

    if (!keyState[dir]) {
        keyState[dir] = true;
        updateKeyDisplay();
        if (manualMode) sendKeys();
    }
});

document.addEventListener('keyup', e => {
    const dir = keyMap[e.key];
    if (!dir) return;

    e.preventDefault();

    if (keyState[dir]) {
        keyState[dir] = false;
        updateKeyDisplay();
        if (manualMode) sendKeys();
    }
});

window.addEventListener('blur', () => {
    Object.keys(keyState).forEach(k => keyState[k] = false);
    updateKeyDisplay();
    if (manualMode) sendKeys();
});

setInterval(() => {
    if (manualMode && Object.values(keyState).some(Boolean)) {
        sendKeys();
    }
}, 150);


// =========================================================
// LOGS
// =========================================================
const MAX_LOGS = 150;
const logs = [];

function ts() {
    return new Date().toTimeString().slice(0, 8);
}

function addLog(msg, cls) {
    logs.push({
        ts: ts(),
        msg: msg,
        cls: cls || ''
    });

    if (logs.length > MAX_LOGS) {
        logs.shift();
    }

    renderLogs();
}

function clearLogs() {
    logs.length = 0;
    renderLogs();
}

function renderLogs() {
    const el = document.getElementById('bot-logs');
    if (!el) return;

    if (logs.length === 0) {
        el.innerHTML = '<div class="log-line">No logs yet.</div>';
        return;
    }

    el.innerHTML = logs.slice().reverse().map(e => {
        return '<div class="log-line log-' + esc(e.cls) + '">' +
            '<span class="log-ts">' + esc(e.ts) + '</span>' +
            esc(e.msg) +
            '</div>';
    }).join('');
}


// =========================================================
// STATUS POLLING + DEBUG
// =========================================================
let prevRunning = null;
let prevManual = null;
let prevStop = null;
let prevSignState = null;
let prevLaneDetected = null;
let prevDetectionSig = '';
let pollCounter = 0;

function renderRuntime(data) {
    renderDebugPanel('runtime-debug', [
        ['hostname', data.hostname],
        ['running', data.running],
        ['manual_mode', data.manual_mode],
        ['camera_ready', data.camera_ready],
        ['wheels_ready', data.wheels_ready],
        ['agent_ready', data.agent_ready],
        ['last_frame_age_sec', data.last_frame_age_sec],
        ['last_error', data.last_error || ''],
        ['last_pwm.left', data.last_pwm ? data.last_pwm.left : null],
        ['last_pwm.right', data.last_pwm ? data.last_pwm.right : null],
        ['stopped_by_detection', data.stopped_by_detection],
        ['stop_reason', data.stop_reason || ''],
    ]);
}

function renderLane(data) {
    const lane = data.lane_debug || {};
    const params = lane.params || {};

    setChip(
        'lane-state-chip',
        lane.lane_detected ? 'LANE OK' : 'NO LANE',
        lane.lane_detected ? 'ok' : 'bad'
    );

    renderDebugPanel('lane-debug', [
        ['available', lane.available],
        ['frame_count', lane.frame_count],
        ['lane_detected', lane.lane_detected],
        ['total_lane_pixels', lane.total_lane_pixels],
        ['lateral_error', lane.lateral_error],
        ['slice_ys', lane.slice_ys || []],
        ['yellow_xs', lane.yellow_xs || []],
        ['white_xs', lane.white_xs || []],
        ['is_curve', lane.is_curve],
        ['curve_dir', lane.curve_dir],
        ['base_speed', params.base_speed],
        ['curve_speed', params.curve_speed],
        ['p_gain', params.p_gain],
        ['d_gain', params.d_gain],
        ['max_steer', params.max_steer],
        ['curve_threshold', params.curve_threshold],
        ['detection_threshold', params.detection_threshold],
        ['lane_half_width', params.lane_half_width],
    ]);
}

function renderSign(data) {
    const signState = data.sign_state || 'NONE';
    const dbg = data.sign_debug || {};

    setChip('sign-state-chip', signState, signState === 'MOVING' ? 'ok' : 'warn');

    const rows = [
        ['sign_state', signState],
    ];

    Object.keys(dbg).sort().forEach(k => {
        rows.push([k, dbg[k]]);
    });

    renderDebugPanel('sign-debug', rows);
}

function renderDetections(data) {
    const dets = data.detections || [];

    const count = document.getElementById('det-count');
    count.textContent = dets.length ? dets.length + ' found' : '0 found';

    const list = document.getElementById('detections');

    if (!dets.length) {
        list.innerHTML = '<div class="empty-state">No detections</div>';
        return;
    }

    list.innerHTML = dets.map(d => {
        return '<div class="det-row">' +
            '<span class="det-class">' + esc(d.class) + '</span>' +
            '<span class="det-score">' + esc(fmt(d.score)) + '</span>' +
            '<span class="det-bbox">' +
                'bbox=' + esc(fmt(d.bbox)) +
                ' area=' + esc(fmt(d.area)) +
                ' center=' + esc(fmt(d.center)) +
            '</span>' +
            '</div>';
    }).join('');
}

function updateStopBanner(data) {
    const banner = document.getElementById('stop-banner');
    const stream = document.getElementById('stream-img');

    if (data.stopped_by_detection) {
        banner.textContent = 'STOPPED — ' + (data.stop_reason || 'detection');
        banner.classList.add('active');
        stream.classList.add('stopped');
    } else {
        banner.classList.remove('active');
        stream.classList.remove('stopped');
    }
}

function statusLogs(data) {
    if (prevRunning !== null && prevRunning !== data.running) {
        addLog(data.running ? '[Server] RUNNING' : '[Server] STOPPED', data.running ? 'ok' : 'warn');
    }
    prevRunning = data.running;

    if (prevManual !== null && prevManual !== data.manual_mode) {
        addLog('[Server] manual_mode=' + data.manual_mode, 'info');
    }
    prevManual = data.manual_mode;

    if (prevStop !== null && prevStop !== data.stopped_by_detection) {
        if (data.stopped_by_detection) {
            addLog('[STOP] ' + (data.stop_reason || 'stopped by detection'), 'warn');
        } else {
            addLog('[STOP] released', 'ok');
        }
    }
    prevStop = data.stopped_by_detection;

    if (prevSignState !== null && prevSignState !== data.sign_state) {
        addLog('[SignFSM] ' + prevSignState + ' -> ' + data.sign_state, 'state');
    }
    prevSignState = data.sign_state;

    const lane = data.lane_debug || {};
    if (prevLaneDetected !== null && prevLaneDetected !== lane.lane_detected) {
        addLog('[Lane] lane_detected=' + lane.lane_detected, lane.lane_detected ? 'ok' : 'warn');
    }
    prevLaneDetected = lane.lane_detected;

    const dets = data.detections || [];
    const sig = dets.map(d => d.class + ':' + d.score + ':' + d.area).join('|');

    if (sig !== prevDetectionSig) {
        if (dets.length) {
            dets.forEach(d => {
                addLog(
                    '[Detection] ' + d.class +
                    ' score=' + fmt(d.score) +
                    ' area=' + fmt(d.area) +
                    ' bbox=' + fmt(d.bbox),
                    'info'
                );
            });
        }
    }

    prevDetectionSig = sig;
}

function consoleDebug(data) {
    pollCounter++;

    if (pollCounter % 5 !== 0) return;

    const lane = data.lane_debug || {};

    console.log('[STATUS]', data);

    console.table({
        running: data.running,
        manual_mode: data.manual_mode,
        camera_ready: data.camera_ready,
        wheels_ready: data.wheels_ready,
        agent_ready: data.agent_ready,
        stopped_by_detection: data.stopped_by_detection,
        stop_reason: data.stop_reason,
        sign_state: data.sign_state,
        pwm_left: data.last_pwm ? data.last_pwm.left : null,
        pwm_right: data.last_pwm ? data.last_pwm.right : null,
    });

    console.table({
        frame_count: lane.frame_count,
        lane_detected: lane.lane_detected,
        pixels: lane.total_lane_pixels,
        error: lane.lateral_error,
        is_curve: lane.is_curve,
        curve_dir: lane.curve_dir,
        slice_ys: (lane.slice_ys || []).join(', '),
        yellow_xs: (lane.yellow_xs || []).join(', '),
        white_xs: (lane.white_xs || []).join(', '),
    });

    if (data.detections && data.detections.length) {
        console.table(data.detections);
    }
}

async function pollStatus() {
    try {
        const data = await fetch('/status').then(r => r.json());
        console.log(data)
        setRunningUI(data.running);
        manualMode = !!data.manual_mode;

        renderRuntime(data);
        renderLane(data);
        renderSign(data);
        renderDetections(data);
        updateStopBanner(data);
        statusLogs(data);
        consoleDebug(data);

    } catch (e) {
        addLog('[Status] fetch failed: ' + e, 'error');
    }
}

setInterval(pollStatus, 300);
pollStatus();
"""


SIGN_DETECTION_TEMPLATE = render_template(
    "Final Project - Sign Detection",
    "{{ hostname }} — Sign Detection Debug",
    CONTENT,
    extra_css=_EXTRA_CSS,
    extra_js=EXTRA_JS,
)