from .base import render_template

_EXTRA_CSS = '''
.key-display {
    display: grid;
    grid-template-areas: ". up ." "left down right";
    gap: 4px;
    justify-content: center;
    margin: 8px 0 4px;
}
.key-box {
    width: 32px; height: 32px;
    display: flex; align-items: center; justify-content: center;
    border: 1px solid var(--border-color);
    border-radius: 4px;
    font-size: 13px; font-weight: 700;
    color: var(--text-muted);
    background: var(--bg-sidebar);
    transition: background 0.1s, border-color 0.1s, color 0.1s;
}
.key-box.active { background: rgba(63,185,80,0.2); border-color: var(--accent-green); color: var(--accent-green); }
.key-up    { grid-area: up; }
.key-down  { grid-area: down; }
.key-left  { grid-area: left; }
.key-right { grid-area: right; }

.video-wrapper { position: relative; display: inline-block; line-height: 0; }
.state-banner {
    display: none;
    position: absolute;
    top: 10px;
    left: 50%;
    transform: translateX(-50%);
    background: rgba(31,111,235,0.88);
    color: #fff;
    text-align: center;
    font-size: 13px;
    font-weight: 700;
    padding: 6px 16px;
    border-radius: 4px;
    letter-spacing: 0.5px;
    white-space: nowrap;
    z-index: 10;
    pointer-events: none;
}
.state-banner.active { display: block; }
.state-banner.stop   { background: rgba(192,57,43,0.88); }
.state-banner.yield  { background: rgba(210,153,34,0.88); }
.state-banner.turn   { background: rgba(31,111,235,0.88); }
.state-banner.wait   { background: rgba(160,90,40,0.88); }

.stream.alert { outline: 3px solid #c0392b; }

.fsm-row {
    display: flex;
    justify-content: space-between;
    padding: 6px 0;
    border-bottom: 1px solid var(--border-color);
    font-size: 12px;
}
.fsm-row:last-child { border-bottom: none; }
.fsm-key { color: var(--text-secondary); }
.fsm-val {
    color: var(--text-primary);
    font-family: monospace;
    font-weight: 600;
    font-variant-numeric: tabular-nums;
}
.fsm-pill {
    padding: 2px 8px;
    border-radius: 10px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.5px;
    background: var(--bg-sidebar);
    border: 1px solid var(--border-color);
}
.pill-lane    { color: var(--accent-green);  border-color: rgba(63,185,80,0.4); }
.pill-stop    { color: var(--accent-red);    border-color: rgba(248,81,73,0.4); }
.pill-yield   { color: var(--accent-orange); border-color: rgba(210,153,34,0.4); }
.pill-turn    { color: var(--accent-blue);   border-color: rgba(31,111,235,0.4); }
.pill-wait    { color: var(--accent-purple); border-color: rgba(163,113,247,0.4); }
.pill-obst    { color: var(--accent-red);    border-color: rgba(248,81,73,0.4); }
.pill-init    { color: var(--text-muted); }
'''

_CONTENT = '''
    <div class="container">
        <div class="video-section">
            <div class="video-wrapper">
                <div id="state-banner" class="state-banner"></div>
                <img src="{{ url_for('video') }}" id="stream-img" class="stream">
            </div>
        </div>

        <div class="controls-section">
            <div class="card">
                <div class="card-header">Drive Control</div>
                <div style="display:flex;align-items:center;gap:16px;margin-bottom:12px">
                    <span id="run-indicator" style="display:inline-block;width:14px;height:14px;border-radius:50%;background:#e74c3c;flex-shrink:0"></span>
                    <span id="run-label" style="font-size:14px;font-weight:600;color:var(--text-secondary)">STOPPED</span>
                </div>
                <div style="display:flex;gap:10px;margin-bottom:8px">
                    <button onclick="driveStart()" class="button success" style="flex:1">Start</button>
                    <button onclick="driveStop()"  class="button" style="flex:1;background:var(--accent-orange,#e67e22)">Stop</button>
                </div>
                {% if virtual %}
                <div style="display:flex;gap:10px;margin-bottom:8px">
                    <button id="mode-btn" onclick="toggleMode()" class="button" style="flex:1;background:#555">Manual</button>
                    <button onclick="resetPosition()" class="button" style="flex:1;background:#444">Reset</button>
                </div>
                {% else %}
                <div style="margin-bottom:8px">
                    <button id="mode-btn" onclick="toggleMode()" class="button" style="width:100%;background:#555">Manual</button>
                </div>
                {% endif %}
                <div id="key-panel" style="display:none">
                    <div class="key-display">
                        <div class="key-box key-up"    id="key-up">&#9650;</div>
                        <div class="key-box key-left"  id="key-left">&#9664;</div>
                        <div class="key-box key-down"  id="key-down">&#9660;</div>
                        <div class="key-box key-right" id="key-right">&#9654;</div>
                    </div>
                    <p style="text-align:center;font-size:11px;color:var(--text-muted);margin:4px 0 0">Arrow keys or WASD</p>
                </div>
            </div>

            {% if virtual %}
            <div class="card">
                <div class="card-header">Map</div>
                <button id="scene-btn" onclick="switchScene()" class="button" style="width:100%;background:#446">Switch to Sign-Test Map</button>
            </div>

            <div class="card">
                <div class="card-header">Scene Objects</div>
                <div style="display:flex;gap:10px">
                    <button onclick="removeObjects('duckie')" class="button" style="flex:1;background:#555">Clear Ducks</button>
                    <button onclick="removeObjects('vehicle')" class="button" style="flex:1;background:#555">Clear Cars</button>
                </div>
            </div>
            {% endif %}

            <div class="card">
                <div class="card-header">
                    FSM Status
                    <span id="fsm-state-pill" class="fsm-pill pill-init">INIT</span>
                </div>
                <div id="fsmTable">
                    <div class="fsm-row"><span class="fsm-key">Last sign</span>      <span class="fsm-val" id="fsm-last-sign">—</span></div>
                    <div class="fsm-row"><span class="fsm-key">Chosen maneuver</span><span class="fsm-val" id="fsm-maneuver">—</span></div>
                    <div class="fsm-row"><span class="fsm-key">Tags in frame</span> <span class="fsm-val" id="fsm-tags">0</span></div>
                    <div class="fsm-row"><span class="fsm-key">Obstacle</span>      <span class="fsm-val" id="fsm-obstacle">no</span></div>
                    <div class="fsm-row"><span class="fsm-key">Current map</span>   <span class="fsm-val" id="fsm-scene">project</span></div>
                </div>
            </div>

            <div class="card">
                <div class="card-header">Send Command</div>
                <div style="display:flex;flex-direction:column;gap:8px;">
                    <div style="display:flex;gap:6px;">
                        <input id="cmdKey" type="text" placeholder="key"
                            style="flex:1;padding:6px 8px;background:var(--bg-sidebar);
                                   border:1px solid var(--border-color);border-radius:4px;
                                   color:var(--text-primary);font-size:13px;">
                        <input id="cmdValue" type="text" placeholder="value"
                            style="flex:2;padding:6px 8px;background:var(--bg-sidebar);
                                   border:1px solid var(--border-color);border-radius:4px;
                                   color:var(--text-primary);font-size:13px;">
                    </div>
                    <button class="button" onclick="sendCommand()">Send</button>
                    <div id="cmdStatus" class="status"></div>
                </div>
            </div>
        </div>
    </div>
'''

_EXTRA_JS = '''
    function setRunningUI(isRunning) {
        document.getElementById('run-indicator').style.background = isRunning ? '#2ecc71' : '#e74c3c';
        const label = document.getElementById('run-label');
        label.textContent = isRunning ? 'RUNNING' : 'STOPPED';
        label.style.color = isRunning ? '#2ecc71' : 'var(--text-secondary)';
    }

    function driveStart()    { postJSON('/start', {}).then(() => setRunningUI(true)); }
    function driveStop()     { postJSON('/stop',  {}).then(() => setRunningUI(false)); }
    function removeObjects(f){ postJSON('/remove_objects', {filter: f}); }
    function resetPosition() { postJSON('/reset', {}).then(d => { if (d && d.running !== undefined) setRunningUI(d.running); }); }

    let _currentScene = 'project';
    function switchScene() {
        const target = _currentScene === 'project' ? 'project_signs' : 'project';
        postJSON('/switch_scene', {scene: target}).then(data => {
            if (data && data.scene) _currentScene = data.scene;
            _updateSceneBtn();
        });
    }
    function _updateSceneBtn() {
        const btn = document.getElementById('scene-btn');
        if (!btn) return;
        btn.textContent = (_currentScene === 'project') ? 'Switch to Sign-Test Map'
                                                       : 'Switch to Main Map';
        const lbl = document.getElementById('fsm-scene');
        if (lbl) lbl.textContent = _currentScene;
    }

    let _manualMode = false;
    const keyState = {up:false, down:false, left:false, right:false};
    const keyMap = {
        'ArrowUp':'up','w':'up','W':'up',
        'ArrowDown':'down','s':'down','S':'down',
        'ArrowLeft':'left','a':'left','A':'left',
        'ArrowRight':'right','d':'right','D':'right',
    };
    function updateKeyDisplay() {
        for (const [k,a] of Object.entries(keyState)) {
            const el = document.getElementById('key-'+k);
            if (el) el.classList.toggle('active', a);
        }
    }
    function sendKeys() {
        fetch('/keys', {method:'POST', headers:{'Content-Type':'application/json'},
                        body: JSON.stringify(keyState)}).catch(()=>{});
    }
    function toggleMode() {
        _manualMode = !_manualMode;
        postJSON('/set_mode', {mode: _manualMode ? 'manual' : 'auto'});
        const btn = document.getElementById('mode-btn');
        const panel = document.getElementById('key-panel');
        if (btn)   btn.textContent = _manualMode ? 'Auto' : 'Manual';
        if (panel) panel.style.display = _manualMode ? 'block' : 'none';
    }

    document.addEventListener('keydown', e => {
        const dir = keyMap[e.key];
        if (dir && !keyState[dir]) { e.preventDefault(); keyState[dir] = true; updateKeyDisplay(); if (_manualMode) sendKeys(); }
    });
    document.addEventListener('keyup', e => {
        const dir = keyMap[e.key];
        if (dir && keyState[dir]) { e.preventDefault(); keyState[dir] = false; updateKeyDisplay(); if (_manualMode) sendKeys(); }
    });
    window.addEventListener('blur', () => {
        Object.keys(keyState).forEach(k => keyState[k] = false);
        updateKeyDisplay(); if (_manualMode) sendKeys();
    });
    setInterval(() => { if (_manualMode && Object.values(keyState).some(Boolean)) sendKeys(); }, 150);

    function sendCommand() {
        const key   = document.getElementById('cmdKey').value.trim();
        const value = document.getElementById('cmdValue').value.trim();
        if (!key) { showStatus('cmdStatus', 'Key cannot be empty', 'error'); return; }
        postJSON('/command', {key, value})
          .then(r => showStatus('cmdStatus', r.status === 'ok' ? 'Sent' : (r.message||'Error'),
                                              r.status === 'ok' ? 'success' : 'error'))
          .catch(e => showStatus('cmdStatus', 'Error: ' + e, 'error'));
    }
    document.getElementById('cmdValue').addEventListener('keydown', e => {
        if (e.key === 'Enter') sendCommand();
    });

    const STATE_PILL_CLASS = {
        'LANE_FOLLOW':'pill-lane','APPROACH_SIGN':'pill-lane','STOP_HOLD':'pill-stop',
        'YIELD_HOLD':'pill-yield','EXECUTE_TURN':'pill-turn','WAIT_RIGHT_OF_WAY':'pill-wait',
        'OBSTACLE_STOP':'pill-obst','DONE':'pill-init','INIT':'pill-init',
    };

    async function pollStatus() {
        try {
            const data = await fetch('/status').then(r => r.json());
            setRunningUI(!!data.running);

            if (data.current_scene && data.current_scene !== _currentScene) {
                _currentScene = data.current_scene;
                _updateSceneBtn();
            }

            const state = data.state || 'INIT';
            const pill  = document.getElementById('fsm-state-pill');
            pill.textContent = state;
            pill.className = 'fsm-pill ' + (STATE_PILL_CLASS[state] || 'pill-init');

            const banner = document.getElementById('state-banner');
            const stream = document.getElementById('stream-img');
            banner.classList.remove('active','stop','yield','turn','wait');
            stream.classList.remove('alert');
            if (state === 'STOP_HOLD' || state === 'OBSTACLE_STOP') {
                banner.textContent = (state === 'OBSTACLE_STOP') ? 'OBSTACLE — HALT' : 'STOP SIGN — HOLDING';
                banner.classList.add('active','stop');
                stream.classList.add('alert');
            } else if (state === 'YIELD_HOLD') {
                banner.textContent = 'YIELD'; banner.classList.add('active','yield');
            } else if (state === 'EXECUTE_TURN') {
                const m = data.chosen_maneuver ? data.chosen_maneuver.toUpperCase() : 'TURN';
                banner.textContent = 'TURNING — ' + m; banner.classList.add('active','turn');
            } else if (state === 'WAIT_RIGHT_OF_WAY') {
                banner.textContent = 'WAITING — RIGHT OF WAY'; banner.classList.add('active','wait');
            }

            document.getElementById('fsm-last-sign').textContent = data.last_sign       ? data.last_sign       : '—';
            document.getElementById('fsm-maneuver') .textContent = data.chosen_maneuver ? data.chosen_maneuver : '—';
            document.getElementById('fsm-tags')     .textContent = (data.tag_count ?? 0);
            document.getElementById('fsm-obstacle') .textContent = data.obstacle ? 'YES' : 'no';
            document.getElementById('fsm-scene')    .textContent = data.current_scene || _currentScene;

            if (typeof data.manual_mode === 'boolean' && data.manual_mode !== _manualMode) {
                _manualMode = data.manual_mode;
                const btn = document.getElementById('mode-btn');
                const panel = document.getElementById('key-panel');
                if (btn)   btn.textContent = _manualMode ? 'Auto' : 'Manual';
                if (panel) panel.style.display = _manualMode ? 'block' : 'none';
            }
        } catch (e) {}
    }

    setInterval(pollStatus, 300);
    pollStatus();
'''


def get_template(title='Project — Traffic Signs', subtitle='Simulation'):
    return render_template(
        title=title,
        subtitle=subtitle,
        content_html=_CONTENT,
        extra_css=_EXTRA_CSS,
        extra_js=_EXTRA_JS,
    )
