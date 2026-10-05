// Graphic Ink: sine-displaced lines, dots, or ordered dither whose weight follows
// the luminance of an image, video, or animated canvas. Two-colour output.

const canvas = document.getElementById('stage');
const gl = canvas.getContext('webgl2', { antialias: false, alpha: false });
if (!gl) document.body.textContent = 'WebGL2 is required.';

// ---------- Shader ----------

const VERT = `#version 300 es
in vec2 aPos;
void main() { gl_Position = vec4(aPos, 0.0, 1.0); }`;

const FRAG = `#version 300 es
precision highp float;
uniform sampler2D uTex;
uniform vec2 uRes, uScale;
uniform float uSpacing, uAmp, uK, uShear, uPhase, uAngle;
uniform float uContrast, uBrightness, uMin, uMax, uLod, uInvert;
uniform int uMode; // 0 lines, 1 dots, 2 dither
uniform vec3 uInk, uPaper;
out vec4 outColor;

float c, s;

// Ink weight (0..1) for the source at a point in the rotated frame.
float weightAt(vec2 qc) {
  vec2 pc = vec2(c * qc.x - s * qc.y, s * qc.x + c * qc.y);
  float lum = dot(textureLod(uTex, pc / uRes * uScale + 0.5, uLod).rgb, vec3(0.299, 0.587, 0.114));
  lum = clamp((lum - 0.5) * uContrast + 0.5 + uBrightness, 0.0, 1.0);
  lum = mix(lum, 1.0 - lum, uInvert);
  return mix(uMin, uMax, 1.0 - lum);
}

// Ordered-dither threshold from an 8x8 Bayer matrix.
float bayer2(vec2 a) { a = floor(a); return fract(a.x * 0.5 + a.y * a.y * 0.75); }
float bayer4(vec2 a) { return bayer2(0.5 * a) * 0.25 + bayer2(a); }
float bayer8(vec2 a) { return bayer4(0.5 * a) * 0.25 + bayer2(a); }

void main() {
  // Pixel position in the rotated frame: rows run along q.x.
  vec2 p = gl_FragCoord.xy - 0.5 * uRes;
  c = cos(uAngle);
  s = sin(uAngle);
  vec2 q = vec2(c * p.x + s * p.y, -s * p.x + c * p.y);

  // Every pattern is laid out in the warped frame (q.x, y), y = q.y + wave(q).
  float arg = q.x * uK + q.y * uShear + uPhase;
  float y = q.y + uAmp * sin(arg);
  vec2 grad = vec2(uAmp * uK * cos(arg), 1.0 + uAmp * uShear * cos(arg));

  vec2 here = p / uRes * uScale + 0.5;
  float inside = step(0.0, here.x) * step(here.x, 1.0) * step(0.0, here.y) * step(here.y, 1.0);

  float a;
  if (uMode == 0) {
    // Offset from the nearest line centre, measured across the line pitch.
    float dy = y - (floor(y / uSpacing) + 0.5) * uSpacing;
    // Sample the source at the line centre so each line stays symmetric.
    float width = weightAt(vec2(q.x, q.y - dy / grad.y)) * uSpacing;
    // Weight is a share of the pitch, so tone holds where the wave bunches lines
    // together; dividing by the gradient turns the edge into a 1px ramp.
    a = clamp((width * 0.5 - abs(dy)) / length(grad) + 0.5, 0.0, 1.0) * smoothstep(0.0, 1.0, width);
  } else {
    // Grid cell in the warped frame, and its centre mapped back to the rotated frame.
    vec2 cell = floor(vec2(q.x, y) / uSpacing);
    vec2 cw = (cell + 0.5) * uSpacing;
    float qy = q.y - (y - cw.y) / grad.y;
    vec2 qc = vec2(cw.x, cw.y - uAmp * sin(cw.x * uK + qy * uShear + uPhase));
    float weight = weightAt(qc);
    if (uMode == 1) {
      // Round dot whose area tracks the weight; it fills the cell at weight 1.
      float r = sqrt(weight) * 0.7072 * uSpacing;
      a = clamp(r - length(q - qc) + 0.5, 0.0, 1.0) * smoothstep(0.0, 1.0, r);
    } else {
      a = step(bayer8(cell) + 1.0 / 128.0, weight);
    }
  }
  outColor = vec4(mix(uPaper, uInk, a * inside), 1.0);
}`;

function compile(type, src) {
  const sh = gl.createShader(type);
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(sh));
  return sh;
}

const program = gl.createProgram();
gl.attachShader(program, compile(gl.VERTEX_SHADER, VERT));
gl.attachShader(program, compile(gl.FRAGMENT_SHADER, FRAG));
gl.linkProgram(program);
if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));
gl.useProgram(program);

const U = {};
for (let i = 0; i < gl.getProgramParameter(program, gl.ACTIVE_UNIFORMS); i++) {
  const { name } = gl.getActiveUniform(program, i);
  U[name] = gl.getUniformLocation(program, name);
}

gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
const aPos = gl.getAttribLocation(program, 'aPos');
gl.enableVertexAttribArray(aPos);
gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);

gl.bindTexture(gl.TEXTURE_2D, gl.createTexture());
gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);

// ---------- Parameters & controls ----------

const SLIDERS = {
  lines: [
    { id: 'spacing', label: 'Spacing', min: 2, max: 40, step: 0.5, value: 9, unit: 'px' },
    { id: 'min', label: 'Min weight', min: 0, max: 0.5, step: 0.01, value: 0.04 },
    { id: 'max', label: 'Max weight', min: 0.1, max: 1.2, step: 0.01, value: 0.95 },
    { id: 'angle', label: 'Angle', min: -90, max: 90, step: 1, value: 0, unit: '°' },
  ],
  wave: [
    { id: 'amp', label: 'Amplitude', min: 0, max: 40, step: 0.5, value: 6, unit: 'px' },
    { id: 'wavelength', label: 'Wavelength', min: 20, max: 600, step: 1, value: 120, unit: 'px' },
    { id: 'speed', label: 'Speed', min: -4, max: 4, step: 0.1, value: 1 },
    { id: 'stagger', label: 'Stagger', min: 0, max: 1.5, step: 0.01, value: 0.15 },
  ],
  tone: [
    { id: 'contrast', label: 'Contrast', min: 0.2, max: 3, step: 0.05, value: 1.2 },
    { id: 'brightness', label: 'Brightness', min: -0.5, max: 0.5, step: 0.01, value: 0 },
  ],
};

const PALETTES = [
  { name: 'Ink', ink: '#111111', paper: '#f2efe6' },
  { name: 'Navy', ink: '#0b1f4b', paper: '#e9eef8' },
  { name: 'Oxblood', ink: '#5a0f14', paper: '#f6e9e2' },
  { name: 'Forest', ink: '#0f3d2e', paper: '#e8f1e4' },
  { name: 'Phosphor', ink: '#7cffb2', paper: '#06130c' },
  { name: 'Amber', ink: '#ffb000', paper: '#140c00' },
];

const params = { mode: 'lines', invert: false, fit: 'cover', ink: PALETTES[0].ink, paper: PALETTES[0].paper };

for (const [group, defs] of Object.entries(SLIDERS)) {
  const host = document.getElementById(`sliders-${group}`);
  for (const def of defs) {
    params[def.id] = def.value;
    const row = document.createElement('label');
    row.className = 'slider';
    row.innerHTML = `<span>${def.label}</span><output></output>
      <input type="range" min="${def.min}" max="${def.max}" step="${def.step}" value="${def.value}">`;
    const out = row.querySelector('output');
    const input = row.querySelector('input');
    const show = () => { out.textContent = `${+params[def.id].toFixed(2)}${def.unit || ''}`; };
    input.addEventListener('input', () => { params[def.id] = +input.value; show(); });
    show();
    host.append(row);
  }
}

const inkInput = document.getElementById('ink');
const paperInput = document.getElementById('paper');
const swatches = document.getElementById('swatches');

function setColours(ink, paper) {
  params.ink = ink;
  params.paper = paper;
  inkInput.value = ink;
  paperInput.value = paper;
  document.documentElement.style.setProperty('--ink', ink);
  document.documentElement.style.setProperty('--paper', paper);
  for (const b of swatches.children) {
    b.setAttribute('aria-pressed', b.dataset.ink === ink && b.dataset.paper === paper);
  }
}

for (const p of PALETTES) {
  const b = document.createElement('button');
  b.className = 'swatch';
  b.title = p.name;
  b.dataset.ink = p.ink;
  b.dataset.paper = p.paper;
  b.style.background = `linear-gradient(135deg, ${p.ink} 50%, ${p.paper} 50%)`;
  b.addEventListener('click', () => setColours(p.ink, p.paper));
  swatches.append(b);
}
setColours(params.ink, params.paper);

inkInput.addEventListener('input', () => setColours(inkInput.value, params.paper));
paperInput.addEventListener('input', () => setColours(params.ink, paperInput.value));
document.getElementById('swap').addEventListener('click', () => setColours(params.paper, params.ink));
const MODES = ['lines', 'dots', 'dither'];
const modeButtons = document.querySelectorAll('#mode button');
for (const b of modeButtons) {
  b.addEventListener('click', () => {
    params.mode = b.dataset.mode;
    for (const o of modeButtons) o.setAttribute('aria-pressed', o === b);
  });
}

document.getElementById('invert').addEventListener('change', (e) => { params.invert = e.target.checked; });
document.getElementById('fit').addEventListener('change', (e) => { params.fit = e.target.value; });

const panel = document.getElementById('panel');
const toggle = document.getElementById('toggle');
function togglePanel() {
  const collapsed = panel.classList.toggle('collapsed');
  toggle.setAttribute('aria-expanded', !collapsed);
}
toggle.addEventListener('click', togglePanel);
addEventListener('keydown', (e) => {
  if (e.key.toLowerCase() === 'h' && !e.metaKey && !e.ctrlKey && !e.altKey) togglePanel();
});

// ---------- Sources ----------

const MAX_IMAGE = 4096;
const sourceName = document.getElementById('source-name');
const playPause = document.getElementById('playpause');

// source = { el, w, h, live } — live sources are re-uploaded every frame.
let source = null;
let needsUpload = true;
let objectUrl = null;

// Built-in animated element: soft blobs drifting on black.
const demo = document.createElement('canvas');
demo.width = 640;
demo.height = 400;
const dctx = demo.getContext('2d');

function drawDemo(t) {
  const { width: w, height: h } = demo;
  dctx.globalCompositeOperation = 'source-over';
  dctx.fillStyle = '#000';
  dctx.fillRect(0, 0, w, h);
  dctx.globalCompositeOperation = 'lighter';
  for (let i = 0; i < 4; i++) {
    const x = w * (0.5 + 0.36 * Math.sin(t * (0.31 + i * 0.07) + i * 2.1));
    const y = h * (0.5 + 0.34 * Math.cos(t * (0.23 + i * 0.05) + i * 1.3));
    const r = h * (0.32 + 0.1 * Math.sin(t * 0.4 + i));
    const g = dctx.createRadialGradient(x, y, 0, x, y, r);
    g.addColorStop(0, 'rgba(255,255,255,0.9)');
    g.addColorStop(1, 'rgba(255,255,255,0)');
    dctx.fillStyle = g;
    dctx.fillRect(0, 0, w, h);
  }
  // A hard-edged disc so the lines have a crisp shape to resolve.
  dctx.globalCompositeOperation = 'difference';
  dctx.fillStyle = '#fff';
  dctx.beginPath();
  dctx.arc(w * (0.5 + 0.2 * Math.cos(t * 0.5)), h * (0.5 + 0.2 * Math.sin(t * 0.7)), h * 0.16, 0, Math.PI * 2);
  dctx.fill();
}

function setSource(next, name) {
  if (source && source.el instanceof HTMLVideoElement) source.el.pause();
  if (objectUrl && (!next.url || next.url !== objectUrl)) URL.revokeObjectURL(objectUrl);
  objectUrl = next.url || null;
  source = next;
  needsUpload = true;
  sourceName.textContent = name;
  playPause.hidden = !(next.el instanceof HTMLVideoElement);
  playPause.textContent = 'Pause';
}

function useDemo() {
  setSource({ el: demo, w: demo.width, h: demo.height, live: true }, 'Demo animation');
}

function loadFile(file) {
  if (!file) return;
  const url = URL.createObjectURL(file);
  if (file.type.startsWith('video/')) {
    const video = document.createElement('video');
    video.muted = true;
    video.loop = true;
    video.playsInline = true;
    video.src = url;
    const syncLabel = () => { if (source.el === video) playPause.textContent = video.paused ? 'Play' : 'Pause'; };
    video.addEventListener('play', syncLabel);
    video.addEventListener('pause', syncLabel);
    video.addEventListener('loadeddata', () => {
      setSource({ el: video, w: video.videoWidth, h: video.videoHeight, live: true, url }, file.name);
      video.play();
    }, { once: true });
    video.addEventListener('error', () => { sourceName.textContent = `Can't play ${file.name}`; URL.revokeObjectURL(url); }, { once: true });
  } else if (file.type.startsWith('image/')) {
    const img = new Image();
    img.onload = () => {
      // Keep huge images within texture limits.
      const k = Math.min(1, MAX_IMAGE / Math.max(img.naturalWidth, img.naturalHeight));
      let el = img;
      if (k < 1) {
        el = document.createElement('canvas');
        el.width = Math.round(img.naturalWidth * k);
        el.height = Math.round(img.naturalHeight * k);
        el.getContext('2d').drawImage(img, 0, 0, el.width, el.height);
      }
      setSource({ el, w: el.width, h: el.height, live: false, url }, file.name);
    };
    img.onerror = () => { sourceName.textContent = `Can't read ${file.name}`; URL.revokeObjectURL(url); };
    img.src = url;
  } else {
    sourceName.textContent = `Unsupported file: ${file.name}`;
    URL.revokeObjectURL(url);
  }
}

const fileInput = document.getElementById('file');
fileInput.addEventListener('change', () => { loadFile(fileInput.files[0]); fileInput.value = ''; });
document.getElementById('demo').addEventListener('click', useDemo);
playPause.addEventListener('click', () => {
  const v = source.el;
  if (v.paused) v.play(); else v.pause();
});

let dragDepth = 0;
addEventListener('dragenter', (e) => { e.preventDefault(); dragDepth++; document.body.classList.add('dragging'); });
addEventListener('dragleave', () => { if (--dragDepth <= 0) { dragDepth = 0; document.body.classList.remove('dragging'); } });
addEventListener('dragover', (e) => e.preventDefault());
addEventListener('drop', (e) => {
  e.preventDefault();
  dragDepth = 0;
  document.body.classList.remove('dragging');
  loadFile(e.dataTransfer.files[0]);
});

// ---------- Render ----------

const hexToRgb = (hex) => [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255);

let dpr = 1;
function resize() {
  dpr = Math.min(devicePixelRatio || 1, 2);
  const w = Math.round(innerWidth * dpr);
  const h = Math.round(innerHeight * dpr);
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
    gl.viewport(0, 0, w, h);
  }
}
addEventListener('resize', resize);
resize();

let phase = 0;
let last = performance.now();

function render(now = performance.now()) {
  const dt = Math.min((now - last) / 1000, 0.1);
  last = now;
  phase += params.speed * dt;

  if (source.el === demo) drawDemo(now / 1000);
  const ready = !(source.el instanceof HTMLVideoElement) || source.el.readyState >= 2;
  if ((source.live || needsUpload) && ready) {
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, source.el);
    gl.generateMipmap(gl.TEXTURE_2D);
    needsUpload = false;
  }

  // Map canvas space to source UVs for cover / contain.
  const ca = canvas.width / canvas.height;
  const sa = source.w / source.h;
  const fitWidth = (ca > sa) === (params.fit === 'cover');
  const scale = fitWidth ? [1, sa / ca] : [ca / sa, 1];

  // Average the source over roughly one line pitch so thin detail doesn't alias.
  const spacing = params.spacing * dpr;
  const amp = params.amp * dpr;
  const texelsPerPx = (source.w * scale[0]) / canvas.width;
  const lod = Math.log2(Math.max(1, texelsPerPx * spacing * 0.5));
  // Keep the phase shear gentle enough that neighbouring lines never cross.
  const shear = Math.min(params.stagger / spacing, amp > 0 ? 0.8 / amp : Infinity);

  gl.uniform2f(U.uRes, canvas.width, canvas.height);
  gl.uniform2f(U.uScale, scale[0], scale[1]);
  gl.uniform1i(U.uMode, MODES.indexOf(params.mode));
  gl.uniform1f(U.uSpacing, spacing);
  gl.uniform1f(U.uAmp, amp);
  gl.uniform1f(U.uK, (Math.PI * 2) / (params.wavelength * dpr));
  gl.uniform1f(U.uShear, shear);
  gl.uniform1f(U.uPhase, phase);
  gl.uniform1f(U.uAngle, (params.angle * Math.PI) / 180);
  gl.uniform1f(U.uContrast, params.contrast);
  gl.uniform1f(U.uBrightness, params.brightness);
  gl.uniform1f(U.uMin, params.min);
  gl.uniform1f(U.uMax, params.max);
  gl.uniform1f(U.uLod, lod);
  gl.uniform1f(U.uInvert, params.invert ? 1 : 0);
  gl.uniform3fv(U.uInk, hexToRgb(params.ink));
  gl.uniform3fv(U.uPaper, hexToRgb(params.paper));
  gl.drawArrays(gl.TRIANGLES, 0, 3);
}

function loop(now) {
  render(now);
  requestAnimationFrame(loop);
}

// ---------- Export ----------

function download(blob, name) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

document.getElementById('export').addEventListener('click', () => {
  render(); // the drawing buffer is only readable in the same task it was drawn
  canvas.toBlob((blob) => download(blob, 'graphic-ink.png'));
});

// Records the canvas as displayed; MP4 where the browser can encode it, else WebM.
const REC_TYPES = ['video/mp4;codecs=avc1', 'video/mp4', 'video/webm;codecs=vp9', 'video/webm'];
const recType = window.MediaRecorder && REC_TYPES.find((t) => MediaRecorder.isTypeSupported(t));
const recordBtn = document.getElementById('record');
let recorder = null;
recordBtn.hidden = !recType;

recordBtn.addEventListener('click', () => {
  if (recorder) { recorder.stop(); return; }

  const chunks = [];
  const stream = canvas.captureStream(60);
  // Fine lines fall apart at default bitrates, so ask for plenty.
  recorder = new MediaRecorder(stream, { mimeType: recType, videoBitsPerSecond: 20e6 });
  recorder.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };

  const started = performance.now();
  const tick = () => {
    const s = Math.floor((performance.now() - started) / 1000);
    recordBtn.textContent = `Stop ${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
  };
  const timer = setInterval(tick, 250);
  tick();
  recordBtn.classList.add('recording');

  recorder.onstop = () => {
    clearInterval(timer);
    stream.getTracks().forEach((t) => t.stop());
    recorder = null;
    recordBtn.classList.remove('recording');
    recordBtn.textContent = 'Record video';
    download(new Blob(chunks, { type: recType }), `graphic-ink.${recType.includes('mp4') ? 'mp4' : 'webm'}`);
  };
  recorder.start(1000);
});

useDemo();
requestAnimationFrame(loop);
