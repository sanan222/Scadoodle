'use strict';

const API_BASE = '';
const $ = id => document.getElementById(id);

/* ── State ──────────────────────────────────────────────── */
const state = {
  files: [],
  videoURL: null,
  duration: 0,
  selectedFrameIdx: null,
  showingFrame: false,
  videoId: null,
  analyzing: false,
};

/* ── DOM refs ────────────────────────────────────────────── */
const uploadScreen   = $('upload-screen');
const invScreen      = $('investigation-screen');
const dropZone       = $('drop-zone');
const fileInput      = $('file-input');
const fileQueue      = $('file-queue');
const fileList       = $('file-list');
const analyzeBtn     = $('analyze-btn');
const backBtn        = $('back-btn');
const mainVideo      = $('main-video');
const videoPh        = $('video-ph');
const vpVideoLayer   = $('vp-video-layer');
const vpFrameLayer   = $('vp-frame-layer');
const frameImg       = $('frame-img');
const backToVideo    = $('back-to-video');
const frameInfoPill  = $('frame-info-pill');
const frameInfoText  = $('frame-info-text');
const playBtn        = $('play-btn');
const playIcon       = $('play-icon');
const pauseIcon      = $('pause-icon');
const muteBtn        = $('mute-btn');
const volIcon        = $('vol-icon');
const muteIconEl     = $('mute-icon');
const ctrlTime       = $('ctrl-time');
const ctrlDuration   = $('ctrl-duration');
const timelineTrack  = $('timeline-track');
const timelineFill   = $('timeline-fill');
const timelineThumb  = $('timeline-thumb');
const tlEvents       = $('tl-events');
const videoTitle     = $('video-title');
const frameStrip     = $('frame-strip');
const stripCount     = $('strip-count');
const captureBarFill = $('capture-bar-fill');
const captureStatus  = $('capture-status');
const promptInput    = $('prompt-input');
const promptSend     = $('prompt-send');
const aiToast        = $('ai-toast');
const aiToastText    = $('ai-toast-text');
const aiToastFrame   = $('ai-toast-frame');
const aiToastClose   = $('ai-toast-close');

/* ════════════════════════════════════════════════════════════
   FRAME CAPTURER
   ════════════════════════════════════════════════════════════ */
class FrameCapturer {
  constructor({ onFrame, onProgress, onComplete }) {
    this.onFrame    = onFrame;
    this.onProgress = onProgress;
    this.onComplete = onComplete;
    this._video  = document.createElement('video');
    this._canvas = document.createElement('canvas');
    this._ctx    = this._canvas.getContext('2d');
    this._video.muted = true;
    this._video.preload = 'auto';
    this._running = false;
    this._time = 0;
    this._duration = 0;
    this._interval = 0.5; // 2 FPS
    this._video.addEventListener('seeked',         () => this._onSeeked());
    this._video.addEventListener('loadedmetadata', () => this._start());
    this._video.addEventListener('error',          () => this._abort());
  }

  load(src) { this._running = false; this._video.src = src; }

  _start() {
    this._duration = this._video.duration;
    if (!isFinite(this._duration) || this._duration <= 0) { this._abort(); return; }
    const vw = this._video.videoWidth || 640;
    const vh = this._video.videoHeight || 360;
    const scale = Math.min(1, 400 / vw);
    this._canvas.width  = Math.round(vw * scale);
    this._canvas.height = Math.round(vh * scale);
    this._running = true;
    this._time = 0;
    this._video.currentTime = 0;
  }

  _onSeeked() {
    if (!this._running) return;
    this._ctx.drawImage(this._video, 0, 0, this._canvas.width, this._canvas.height);
    const dataURL = this._canvas.toDataURL('image/jpeg', 0.6);
    this.onFrame({ time: this._time, dataURL });
    this.onProgress(Math.min(this._time / this._duration, 1));
    this._time += this._interval;
    if (this._time <= this._duration + 0.01) {
      this._video.currentTime = Math.min(this._time, this._duration);
    } else {
      this._running = false;
      this.onComplete();
    }
  }

  _abort() { this._running = false; this.onComplete(); }
  destroy() { this._running = false; this._video.src = ''; }
}

/* ════════════════════════════════════════════════════════════
   UPLOAD SCREEN
   ════════════════════════════════════════════════════════════ */
['dragenter', 'dragover'].forEach(ev =>
  dropZone.addEventListener(ev, e => { e.preventDefault(); dropZone.classList.add('drag-over'); })
);
['dragleave', 'drop'].forEach(ev =>
  dropZone.addEventListener(ev, e => { e.preventDefault(); dropZone.classList.remove('drag-over'); })
);
dropZone.addEventListener('drop', e => {
  const files = [...e.dataTransfer.files].filter(f => f.type.startsWith('video/'));
  if (files.length) addFiles(files);
});
fileInput.addEventListener('change', e => {
  if (e.target.files.length) addFiles([...e.target.files]);
  fileInput.value = '';
});

function addFiles(arr) {
  arr.forEach(f => {
    if (!state.files.find(x => x.name === f.name && x.size === f.size)) state.files.push(f);
  });
  renderFileList();
}

function renderFileList() {
  if (!state.files.length) { fileQueue.style.display = 'none'; return; }
  fileQueue.style.display = 'block';
  fileList.innerHTML = '';
  state.files.forEach((f, i) => {
    const li = document.createElement('li');
    li.className = 'file-item fade-in';
    li.innerHTML = `
      <div class="file-item-icon"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="23 7 16 12 23 17 23 7"/><rect x="1" y="5" width="15" height="14" rx="2" ry="2"/></svg></div>
      <div class="file-item-info">
        <div class="file-item-name">${esc(f.name)}</div>
        <div class="file-item-size">${fmtBytes(f.size)} · Video</div>
      </div>
      <button class="file-item-remove" data-i="${i}" title="Remove"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg></button>`;
    li.querySelector('.file-item-remove').addEventListener('click', e => {
      e.stopPropagation();
      state.files.splice(parseInt(e.currentTarget.dataset.i), 1);
      renderFileList();
    });
    fileList.appendChild(li);
  });
}

analyzeBtn.addEventListener('click', () => {
  if (state.files.length) openInvestigation(state.files[0]);
});

/* ════════════════════════════════════════════════════════════
   INVESTIGATION SCREEN
   ════════════════════════════════════════════════════════════ */
let capturer = null;
let allFrames = [];

async function openInvestigation(file) {
  uploadScreen.classList.remove('active');
  invScreen.classList.add('active');

  if (state.videoURL) URL.revokeObjectURL(state.videoURL);
  state.videoURL = URL.createObjectURL(file);
  state.videoId = null;
  allFrames = [];
  state.selectedFrameIdx = null;
  state.showingFrame = false;

  videoTitle.textContent = file.name.replace(/\.[^.]+$/, '');

  frameStrip.innerHTML = '';
  stripCount.textContent = '0';
  captureBarFill.style.width = '0%';
  captureStatus.textContent = 'Uploading…';
  showVideoLayer();
  aiToast.style.display = 'none';
  tlEvents.innerHTML = '';

  mainVideo.src = state.videoURL;
  mainVideo.load();
  mainVideo.addEventListener('loadedmetadata', () => {
    state.duration = mainVideo.duration;
    videoPh.style.display = 'none';
    ctrlDuration.textContent = fmtSec(state.duration);
  }, { once: true });

  // Upload video to backend and get server-extracted frames
  const formData = new FormData();
  formData.append('video', file);
  try {
    captureStatus.textContent = 'Uploading to server…';
    const res = await fetch(`${API_BASE}/api/upload`, { method: 'POST', body: formData });
    const data = await res.json();
    if (data.error) {
      captureStatus.textContent = `Error: ${data.error}`;
      return;
    }
    state.videoId = data.video_id;
    captureBarFill.style.width = '50%';
    captureStatus.textContent = `Loading ${data.frame_count} frames…`;

    // Load server-extracted frames into the strip
    const frameNames = data.frames || [];
    const fps = 2;
    for (let i = 0; i < frameNames.length; i++) {
      const url = `${API_BASE}/api/frames/${data.video_id}/${frameNames[i]}`;
      const time = i / fps;
      allFrames.push({ dataURL: url, time });
      addFrameTile({ dataURL: url, time }, i);
      stripCount.textContent = allFrames.length;
      captureBarFill.style.width = (50 + ((i + 1) / frameNames.length) * 50).toFixed(1) + '%';
    }

    captureBarFill.style.width = '100%';
    captureStatus.textContent = `Ready · ${allFrames.length} frames`;

  } catch (e) {
    captureStatus.textContent = `Upload failed: ${e.message}`;
    return;
  }
}

backBtn.addEventListener('click', () => {
  mainVideo.pause();
  invScreen.classList.remove('active');
  uploadScreen.classList.add('active');
});

/* ── Video / Frame layer toggle ──────────────────────────── */
function showVideoLayer() {
  state.showingFrame = false;
  vpFrameLayer.style.display = 'none';
  vpVideoLayer.style.display = 'flex';
}

function showFrameLayer(dataURL, label) {
  state.showingFrame = true;
  mainVideo.pause();
  updatePlayUI(false);
  frameImg.src = dataURL;
  frameInfoText.textContent = label;
  vpVideoLayer.style.display = 'none';
  vpFrameLayer.style.display = 'flex';
}

backToVideo.addEventListener('click', () => {
  showVideoLayer();
  if (state.selectedFrameIdx !== null) {
    const tile = frameStrip.querySelector(`[data-idx="${state.selectedFrameIdx}"]`);
    if (tile) tile.classList.remove('selected');
    state.selectedFrameIdx = null;
  }
});

/* ── Frame tiles ─────────────────────────────────────────── */
function addFrameTile(frame, idx) {
  const tile = document.createElement('div');
  tile.className = 'frame-tile fade-in';
  tile.dataset.idx = idx;
  const img = document.createElement('img');
  img.src = frame.dataURL;
  img.draggable = false;
  const lbl = document.createElement('div');
  lbl.className = 'frame-tile-time';
  lbl.textContent = fmtSec(frame.time);
  tile.appendChild(img);
  tile.appendChild(lbl);
  tile.addEventListener('click', () => selectFrame(idx));
  frameStrip.appendChild(tile);
}

function selectFrame(idx) {
  if (state.selectedFrameIdx !== null) {
    const prev = frameStrip.querySelector(`[data-idx="${state.selectedFrameIdx}"]`);
    if (prev) prev.classList.remove('selected');
  }
  state.selectedFrameIdx = idx;
  const tile = frameStrip.querySelector(`[data-idx="${idx}"]`);
  if (tile) {
    tile.classList.add('selected');
    tile.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  }
  const frame = allFrames[idx];
  showFrameLayer(frame.dataURL, `Frame #${idx + 1} · ${fmtSec(frame.time)}`);

  if (state.duration) {
    mainVideo.currentTime = frame.time;
    updateTimeline(frame.time / state.duration);
  }
}

/* ── Video controls ─────────────────────────────────────── */
function togglePlay() {
  if (state.showingFrame) showVideoLayer();
  if (mainVideo.paused) { mainVideo.play(); updatePlayUI(true); }
  else                  { mainVideo.pause(); updatePlayUI(false); }
}

mainVideo.addEventListener('click', togglePlay);
playBtn.addEventListener('click', togglePlay);
mainVideo.addEventListener('play',  () => updatePlayUI(true));
mainVideo.addEventListener('pause', () => updatePlayUI(false));
mainVideo.addEventListener('ended', () => updatePlayUI(false));

function updatePlayUI(playing) {
  playIcon.style.display  = playing ? 'none'  : 'block';
  pauseIcon.style.display = playing ? 'block' : 'none';
}

mainVideo.addEventListener('timeupdate', () => {
  ctrlTime.textContent = fmtSec(mainVideo.currentTime);
  if (state.duration) updateTimeline(mainVideo.currentTime / state.duration);
});

muteBtn.addEventListener('click', () => {
  mainVideo.muted = !mainVideo.muted;
  volIcon.style.display    = mainVideo.muted ? 'none'  : 'block';
  muteIconEl.style.display = mainVideo.muted ? 'block' : 'none';
});

/* ── Timeline ────────────────────────────────────────────── */
function updateTimeline(ratio) {
  const p = (Math.min(Math.max(ratio, 0), 1) * 100).toFixed(2) + '%';
  timelineFill.style.width = p;
  timelineThumb.style.left = p;
}

let scrubbing = false;
timelineTrack.addEventListener('mousedown', e => { scrubbing = true; scrub(e); });
document.addEventListener('mousemove', e => { if (scrubbing) scrub(e); });
document.addEventListener('mouseup', () => { scrubbing = false; });

function scrub(e) {
  const r = timelineTrack.getBoundingClientRect();
  const ratio = Math.max(0, Math.min(1, (e.clientX - r.left) / r.width));
  if (state.duration) {
    if (state.showingFrame) showVideoLayer();
    mainVideo.currentTime = ratio * state.duration;
    updateTimeline(ratio);
  }
}

/* ── AI Prompt (chat-style) ──────────────────────────────── */
promptInput.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendPrompt(); }
});
promptSend.addEventListener('click', sendPrompt);
aiToastClose.addEventListener('click', () => { aiToast.style.display = 'none'; });

async function sendPrompt() {
  const q = promptInput.value.trim();
  if (!q || state.analyzing) return;
  if (!state.videoId) {
    aiToast.style.display = 'flex';
    aiToastText.textContent = 'Please upload a video first.';
    aiToastFrame.textContent = '';
    return;
  }

  promptInput.value = '';
  state.analyzing = true;
  promptInput.disabled = true;
  promptSend.disabled = true;

  aiToast.style.display = 'flex';
  aiToastText.textContent = 'Analyzing… Gemini is planning the pipeline';
  aiToastFrame.textContent = '';

  try {
    const res = await fetch(`${API_BASE}/api/analyze`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ video_id: state.videoId, prompt: q }),
    });

    const data = await res.json();

    if (data.error) {
      aiToastText.textContent = `Error: ${data.error}`;
      return;
    }

    let displayText = data.task_summary || 'Analysis complete.';
    if (data.frame_id) {
      displayText += ` — detected at frame ${data.frame_id}`;
    }
    aiToastText.textContent = displayText;
    aiToastFrame.textContent = data.frame_name ? `Frame #${data.frame_id}` : '';

    if (data.has_result_image && data.result_image_url) {
      showFrameLayer(data.result_image_url, `AI Result · Frame #${data.frame_id || '?'}`);
    }

    if (data.frame_id && allFrames.length && state.duration) {
      const frameTime = ((data.frame_id - 1) / allFrames.length) * state.duration;
      const dot = document.createElement('div');
      dot.className = 'tl-event tl-event--ai';
      dot.style.left = ((frameTime / state.duration) * 100).toFixed(2) + '%';
      tlEvents.appendChild(dot);

      const closestIdx = allFrames.reduce((best, f, i) =>
        Math.abs(f.time - frameTime) < Math.abs(allFrames[best].time - frameTime) ? i : best, 0);

      const tile = frameStrip.querySelector(`[data-idx="${closestIdx}"]`);
      if (tile) {
        tile.classList.add('selected');
        tile.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
      }
    }

  } catch (e) {
    aiToastText.textContent = `Request failed: ${e.message}`;
  } finally {
    state.analyzing = false;
    promptInput.disabled = false;
    promptSend.disabled = false;
    promptInput.focus();
  }
}

/* ── Keyboard shortcuts ─────────────────────────────────── */
document.addEventListener('keydown', e => {
  if (!invScreen.classList.contains('active')) return;
  if (document.activeElement === promptInput) return;

  if (e.key === ' ')          { e.preventDefault(); togglePlay(); }
  if (e.key === 'ArrowRight') mainVideo.currentTime = Math.min(state.duration, mainVideo.currentTime + 2);
  if (e.key === 'ArrowLeft')  mainVideo.currentTime = Math.max(0, mainVideo.currentTime - 2);
  if (e.key === '/')          { e.preventDefault(); promptInput.focus(); }
  if (e.key === 'Escape' && state.showingFrame) showVideoLayer();

  if (e.key === ']' && state.selectedFrameIdx !== null && state.selectedFrameIdx < allFrames.length - 1)
    selectFrame(state.selectedFrameIdx + 1);
  if (e.key === '[' && state.selectedFrameIdx !== null && state.selectedFrameIdx > 0)
    selectFrame(state.selectedFrameIdx - 1);
});

/* ── Utilities ──────────────────────────────────────────── */
function fmtSec(s) {
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${m}:${String(sec).padStart(2, '0')}`;
}
function fmtBytes(b) {
  if (b < 1024)    return b + ' B';
  if (b < 1024**2) return (b / 1024).toFixed(1) + ' KB';
  if (b < 1024**3) return (b / 1024**2).toFixed(1) + ' MB';
  return (b / 1024**3).toFixed(2) + ' GB';
}
function esc(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}
