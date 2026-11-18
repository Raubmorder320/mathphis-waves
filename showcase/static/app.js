// 3D-эмулятор Финского залива: MapLibre (карта и рельеф всего залива) + слой three.js
// с водой из расчёта shallow_2d и разрезом волн по клику (showcase/section.py).
import * as THREE from "three";
import { City } from "./city.js";

const $ = id => document.getElementById(id);
const TERRAIN_EXAG = 15;            // рельеф суши и дна (MapLibre) ×15
const SURFACE_LIFT = 2;             // м — вода чуть выше нуля, чтобы не мерцала с дном
const MODELS = [
  { id: "linear", name: "Линейная", color: "#f2f5f8" },
  { id: "saint_venant", name: "Сен-Венан", color: "#ff7a2f" },
  { id: "boussinesq", name: "Буссинеск", color: "#2fe0d0" },
];
// Детальные зоны города (showcase/osm_builder.py, flood_map.py).
const CITY_AREAS = [
  { id: "vo", name: "Васильевский остров", bbox: [59.921, 30.205, 59.957, 30.300] },
  { id: "kronstadt", name: "Кронштадт", bbox: [59.982, 29.700, 60.012, 29.800] },
];
const CITY_ZOOM = 13.2;
let city;
const cityState = { area: null, loading: false, level: 0, steps: 0, manual: false, manualLevel: 2.81, saved: null };
const FOCI = [
  { kicker: "Фокус 1", title: "Финский залив", center: [26.0, 59.85], zoom: 6.5, pitch: 50, bearing: 8 },
  { kicker: "Фокус 2", title: "Вход волны · 27° в. д.", center: [27.35, 60.0], zoom: 8.6, pitch: 66, bearing: 78 },
  { kicker: "Фокус 3", title: "Кронштадт и Котлин", center: [29.62, 59.99], zoom: 10.4, pitch: 68, bearing: 84 },
  { kicker: "Фокус 4", title: "Невская губа", center: [29.98, 59.94], zoom: 10.0, pitch: 62, bearing: 96 },
  { kicker: "Фокус 5 · город", title: "Васильевский остров", center: [30.262, 59.9395], zoom: 15.6, pitch: 68, bearing: 70 },
  { kicker: "Фокус 6 · город", title: "Кронштадт", center: [29.772, 59.9935], zoom: 15.6, pitch: 68, bearing: 30 },
];

const state = {
  manifest: null, scenario: null, eta: null, uv: null,
  t: 0, playing: true, speed: 1800, exag: 100, mode: "water",
  section: null, secT: 0, secPlaying: true, prevSection: null,
};

// ---------------------------------------------------------------------------
// Геометрия: долгота/широта ↔ локальные метры слоя three.js
// ---------------------------------------------------------------------------
let origin, meter;      // MercatorCoordinate юго-западного угла и метр в единицах Меркатора
function local(lon, lat) {
  const m = maplibregl.MercatorCoordinate.fromLngLat([lon, lat], 0);
  return [(m.x - origin.x) / meter, -(m.y - origin.y) / meter];
}

// ---------------------------------------------------------------------------
// Данные сценариев
// ---------------------------------------------------------------------------
async function loadManifest() {
  const m = await (await fetch("/data/manifest.json")).json();
  m.wet = m.h.map(row => row.map(v => v > 0));
  state.manifest = m;
  for (const s of m.scenarios) $("scenario").append(new Option(s.title, s.id));
  $("scenario").value = "front";
  return m;
}

async function loadScenario(id) {
  const m = state.manifest;
  const s = m.scenarios.find(x => x.id === id);
  const [e, uv] = await Promise.all([
    fetch(`/data/${id}_eta.bin`).then(r => r.arrayBuffer()),
    fetch(`/data/${id}_uv.bin`).then(r => r.arrayBuffer()),
  ]);
  state.scenario = s;
  state.eta = new Int16Array(e);
  state.uv = new Int16Array(uv);
  state.etaPeak = Math.max(...s.eta_max.flat().map(Math.abs), 0.05);
  state.etaRef = Math.max(1.2 * s.a, 0.3);    // гребень входной волны — полная подсветка
  // Фронт волны (5 ч) смотрим медленнее, нагон (15 ч) — быстрее.
  state.speed = id === "front" ? 600 : 1800;
  $("speed").value = String(state.speed);
  state.exag = id === "front" ? 300 : 100;
  $("exag").value = state.exag; $("exag-v").textContent = state.exag;
  $("wind-v").textContent = s.wind.toFixed(s.wind % 1 ? 1 : 0);
  $("wind-name").textContent = s.wind ? "западный" : "штиль";
  $("arrow").style.transform = "rotate(90deg)";          // западный ветер дует на восток
  drawGaugeBase();
}

// η сценария в момент t (секунды) в ячейке (iy, ix), с интерполяцией между кадрами.
function frameAt(t) {
  const sc = state.scenario, n = sc.frames, dt = sc.t[1] - sc.t[0];
  const f = Math.min(Math.max(t / dt, 0), n - 1.0001);
  return [Math.floor(f), f - Math.floor(f)];
}

// ---------------------------------------------------------------------------
// Вода: сетка вершин поверх расчётной области
// ---------------------------------------------------------------------------
const NXV = 560, NYV = 240;          // сетка воды на весь залив, ~135 тыс. вершин
const TAPER_KM = 25;                 // к западу от расчётной области η плавно гаснет
let water, sea;

// Маска моря и глубины всего залива (showcase/sea.py): 0 — суша, иначе 1 + 25·√h.
async function loadSea() {
  const meta = await (await fetch("/data/sea.json")).json();
  const img = new Image();
  img.src = "/data/sea.png";
  await img.decode();
  const cv = document.createElement("canvas");
  cv.width = meta.width; cv.height = meta.height;
  const ctx = cv.getContext("2d");
  ctx.drawImage(img, 0, 0);
  const px = ctx.getImageData(0, 0, meta.width, meta.height).data;
  const code = new Uint8Array(meta.width * meta.height);
  for (let i = 0; i < code.length; i++) code[i] = px[4 * i];
  const tex = new THREE.Texture(img);
  tex.minFilter = THREE.LinearFilter; tex.magFilter = THREE.LinearFilter;
  tex.generateMipmaps = false; tex.flipY = false; tex.needsUpdate = true;
  sea = { ...meta, code, tex };
}
function seaCode(lon, lat) {         // код маски в точке (ближайший пиксель)
  const i = Math.round((lon - sea.lon0) / (sea.lon1 - sea.lon0) * (sea.width - 1));
  const j = Math.round((sea.lat1 - lat) / (sea.lat1 - sea.lat0) * (sea.height - 1));
  if (i < 0 || j < 0 || i >= sea.width || j >= sea.height) return 0;
  return sea.code[j * sea.width + i];
}

const WATER_VS = `
  attribute vec3 color; attribute float glow; attribute vec2 seaUV;
  varying vec3 vColor; varying float vGlow; varying vec3 vN; varying vec3 vP; varying vec2 vUV;
  void main() { vColor = color; vGlow = glow; vN = normal; vP = position; vUV = seaUV;
    gl_Position = projectionMatrix * vec4(position, 1.0); }`;

// Вода «как в игре»: длинная волна из расчёта (смещение вершин и нормали) плюс
// декоративное ветровое волнение — 11 октав бегущих волн, каждая со скоростью
// глубокой воды √(g·k); видны только те, что экран разрешает (без мерцания).
const WATER_FS = `
  uniform sampler2D seaTex; uniform vec3 camPos; uniform vec3 sunDir; uniform vec4 cityBox;
  uniform float time; uniform float wind; uniform vec2 windDir; uniform float overlay;
  uniform float octaves;
  varying vec3 vColor; varying float vGlow; varying vec3 vN; varying vec3 vP; varying vec2 vUV;

  float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
  float noise(vec2 p) { vec2 i = floor(p), f = fract(p); vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash(i), hash(i + vec2(1, 0)), u.x), mix(hash(i + vec2(0, 1)), hash(i + vec2(1, 1)), u.x), u.y); }

  void main() {
    float code = texture2D(seaTex, vUV).r * 255.0;
    if (code < 0.7) discard;                                    // суша — берег по подробной маске
    // Внутри зоны детального города воду рисует GPU-затопление (city.js).
    if (vP.x > cityBox.x && vP.x < cityBox.z && vP.y > cityBox.y && vP.y < cityBox.w) discard;
    float depth = pow(max(code - 1.0, 0.0) / 25.0, 2.0);

    vec2 p = vP.xy;
    float mpp = max(length(fwidth(p)), 0.01);                   // метров на пиксель экрана
    float sea = 0.35 + clamp(wind / 25.0, 0.0, 1.0);            // сила волнения от ветра сценария
    vec2 grad = vec2(0.0);
    float crest = 0.0;
    for (int i = 0; i < 11; i++) {
      float fi = float(i);
      if (fi >= octaves) break;                                 // автонастройка качества по FPS
      float L = 6000.0 * pow(0.56, fi);                         // от 6 км до 18 м
      float k = 6.2831853 / L;
      float w = sqrt(9.81 * k);
      float fade = smoothstep(3.0, 9.0, L / mpp);
      // Сила октавы гуляет по морю (пятна ряби), чтобы не было ровных полос.
      float patchy = 0.45 + 1.1 * noise(p / (L * 6.0) + vec2(fi * 7.1, fi * 3.3));
      float slope = (i < 4 ? 0.06 : 0.12) * sea * patchy;
      for (int j = 0; j < 2; j++) {                             // две волны на октаву под разными углами
        float a = fi * 2.399 + float(j) * 1.7;
        vec2 d = normalize(windDir + 0.75 * vec2(cos(a), sin(a)));
        float ph = k * dot(d, p) - w * time + hash(vec2(fi, float(j) + 3.0)) * 6.28;
        grad += d * slope * 0.7 * cos(ph) * fade;
        crest += max(sin(ph), 0.0) * fade * (i > 5 ? 0.08 : 0.0);
      }
    }
    vec3 nLong = normalize(vN);
    vec3 n = normalize(vec3(nLong.xy / max(nLong.z, 0.2) - grad, 1.0));

    vec3 V = normalize(camPos - vP);
    float dist = length(camPos - vP);
    vec3 R = reflect(-V, n);
    vec3 zenith = vec3(0.22, 0.42, 0.72), horizon = vec3(0.78, 0.86, 0.93);
    vec3 sky = mix(horizon, zenith, pow(clamp(R.z, 0.0, 1.0), 0.55));
    float F = 0.02 + 0.98 * pow(1.0 - clamp(dot(n, V), 0.0, 1.0), 5.0);
    float rs = max(dot(R, sunDir), 0.0);
    float sunSpec = pow(rs, 900.0) * 14.0 + pow(rs, 60.0) * 0.6;

    // Тело воды: на мелководье бирюза, на глубине тёмная синь.
    vec3 deep = vec3(0.012, 0.085, 0.165), shallow = vec3(0.07, 0.42, 0.40);
    vec3 body = mix(deep, shallow, exp(-depth / 9.0));
    body += vec3(0.0, 0.10, 0.09) * clamp(crest, 0.0, 1.0);    // просвет гребней
    // Длинная волна из расчёта: гребень светлеет и зеленеет, впадина темнеет.
    float etaN = (1.0 - overlay) * vColor.r;
    body += vec3(0.06, 0.30, 0.26) * clamp(etaN, 0.0, 1.5) - vec3(0.0, 0.05, 0.08) * clamp(-etaN, 0.0, 1.0);
    vec3 c = mix(body, sky, F) + vec3(1.0, 0.92, 0.78) * sunSpec;

    // Пена у берега и там, где длинная волна сильно нелинейна (a/h).
    float foamShore = smoothstep(1.8, 0.2, depth) * smoothstep(0.35, 0.75, noise(p / 35.0 + time * 0.3));
    float foamNl = vGlow * smoothstep(0.45, 0.8, noise(p / 60.0 - time * 0.5) + crest)
                 + smoothstep(0.55, 0.95, etaN) * smoothstep(0.5, 0.85, noise(p / 90.0 + time * 0.4)) * 0.6;
    c = mix(c, vec3(0.93, 0.97, 1.0), clamp(foamShore * 0.7 + foamNl * 0.85, 0.0, 1.0));

    // Режимы «уровень η» и «нелинейность a/h» — цвет расчёта поверх воды.
    c = mix(c, vColor * (0.55 + 0.45 * clamp(dot(n, sunDir), 0.0, 1.0)) + vColor * vGlow * 0.6, overlay);

    // Дымка к горизонту.
    float fog = 1.0 - exp(-dist / 420000.0);
    c = mix(c, horizon * 0.92, fog * 0.85);
    gl_FragColor = vec4(c, mix(0.55, 0.97, smoothstep(0.0, 5.0, depth)));
  }`;

function buildWater(scene) {
  const m = state.manifest, nx = m.nx, ny = m.ny;
  const N = NXV * NYV;
  const pos = new Float32Array(N * 3), col = new Float32Array(N * 3), glow = new Float32Array(N);
  const nrm = new Float32Array(N * 3), uv = new Float32Array(N * 2);
  // Для вершин у расчётной области — четыре ячейки модели и веса только по воде.
  const idx = new Int32Array(N * 4), wts = new Float32Array(N * 4), depth = new Float32Array(N);
  const taper = new Float32Array(N), inModel = [], wet = new Uint8Array(N);
  const b = bbox();
  for (let j = 0; j < NYV; j++) for (let i = 0; i < NXV; i++) {
    const v = j * NXV + i;
    const lon = sea.lon0 + (sea.lon1 - sea.lon0) * i / (NXV - 1);
    const lat = sea.lat0 + (sea.lat1 - sea.lat0) * j / (NYV - 1);
    const [x, y] = local(lon, lat);
    pos[3 * v] = x; pos[3 * v + 1] = y; pos[3 * v + 2] = SURFACE_LIFT;
    nrm[3 * v + 2] = 1;
    uv[2 * v] = (lon - sea.lon0) / (sea.lon1 - sea.lon0);
    uv[2 * v + 1] = (sea.lat1 - lat) / (sea.lat1 - sea.lat0);
    const c = seaCode(lon, lat);
    wet[v] = c > 0;
    depth[v] = c > 0 ? ((c - 1) / 25) ** 2 : 0;
    // Веса по мокрым ячейкам модели; западнее области — затухание, чтобы не было шва.
    const lonM = Math.min(Math.max(lon, b.lon0), b.lon1), latM = Math.min(Math.max(lat, b.lat0), b.lat1);
    const fx = (lonM - b.lon0) / (b.lon1 - b.lon0) * (nx - 1), fy = (latM - b.lat0) / (b.lat1 - b.lat0) * (ny - 1);
    const ix = Math.min(Math.floor(fx), nx - 2), iy = Math.min(Math.floor(fy), ny - 2);
    const a = fx - ix, bb = fy - iy;
    let wsum = 0;
    [[iy, ix, (1 - a) * (1 - bb)], [iy, ix + 1, a * (1 - bb)], [iy + 1, ix, (1 - a) * bb], [iy + 1, ix + 1, a * bb]]
      .forEach(([cy, cx, w], k) => {
        const ok = m.wet[cy][cx];
        idx[4 * v + k] = cy * nx + cx; wts[4 * v + k] = ok ? w : 0;
        wsum += ok ? w : 0;
      });
    if (wsum === 0 && wet[v]) {
      // Узкие проливы и каналы: по подробной маске здесь море, а в сетке модели суша.
      // Берём до четырёх ближайших мокрых ячеек модели в радиусе трёх, с весом 1/d².
      const cand = [];
      const cx0 = Math.round(fx), cy0 = Math.round(fy);
      for (let dy = -3; dy <= 3; dy++) for (let dx = -3; dx <= 3; dx++) {
        const cy = cy0 + dy, cx = cx0 + dx;
        if (cy < 0 || cx < 0 || cy >= ny || cx >= nx || !m.wet[cy][cx]) continue;
        cand.push([(cx - fx) ** 2 + (cy - fy) ** 2 + 0.05, cy, cx]);
      }
      cand.sort((p, q) => p[0] - q[0]);
      cand.slice(0, 4).forEach(([d2, cy, cx], k) => { idx[4 * v + k] = cy * nx + cx; wts[4 * v + k] = 1 / d2; wsum += 1 / d2; });
    }
    for (let k = 0; k < 4; k++) wts[4 * v + k] = wsum > 0 ? wts[4 * v + k] / wsum : 0;
    const kmOut = Math.max(b.lon0 - lon, lon - b.lon1, 0) * 55.6 + Math.max(b.lat0 - lat, lat - b.lat1, 0) * 111;
    taper[v] = wsum > 0 ? Math.max(0, 1 - kmOut / TAPER_KM) : 0;
    if (taper[v] > 0) inModel.push(v);
  }
  // Треугольники — где рядом есть вода; точный берег режет шейдер по маске.
  const tris = [];
  for (let j = 0; j < NYV - 1; j++) for (let i = 0; i < NXV - 1; i++) {
    const a = j * NXV + i, b2 = a + 1, c = a + NXV, d = c + 1;
    if (wet[a] || wet[b2] || wet[c] || wet[d]) tris.push(a, b2, c, b2, d, c);
  }
  const coarse = [];                 // каждая вторая вершина — для режима города
  for (let j = 0; j < NYV - 2; j += 2) for (let i = 0; i < NXV - 2; i += 2) {
    const a = j * NXV + i, b2 = a + 2, c = a + 2 * NXV, d = c + 2;
    if (wet[a] || wet[b2] || wet[c] || wet[d]) coarse.push(a, b2, c, b2, d, c);
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  geo.setAttribute("normal", new THREE.BufferAttribute(nrm, 3));
  geo.setAttribute("color", new THREE.BufferAttribute(col, 3));
  geo.setAttribute("glow", new THREE.BufferAttribute(glow, 1));
  geo.setAttribute("seaUV", new THREE.BufferAttribute(uv, 2));
  geo.setIndex(tris);
  const mat = new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, side: THREE.DoubleSide,
    uniforms: {
      seaTex: { value: sea.tex }, camPos: { value: new THREE.Vector3() }, time: { value: 0 },
      // Низкое солнце на востоке: при взгляде к Петербургу по морю идёт солнечная дорожка.
      sunDir: { value: new THREE.Vector3(0.8, 0.12, 0.26).normalize() },
      wind: { value: 0 }, windDir: { value: new THREE.Vector2(1, 0.12).normalize() }, overlay: { value: 0 },
      octaves: { value: 11 }, cityBox: { value: new THREE.Vector4(1e9, 1e9, -1e9, -1e9) },
    },
    vertexShader: WATER_VS, fragmentShader: WATER_FS,
  });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.frustumCulled = false;
  scene.add(mesh);
  water = { mesh, geo, pos, nrm, col, glow, idx, wts, depth, taper, inModel: Int32Array.from(inModel),
    trisFull: geo.index, trisCoarse: new THREE.BufferAttribute(new Uint32Array(coarse), 1) };
}

const lerp = (a, b, t) => a + (b - a) * t;
function ramp(stops, v) {           // [[x, [r,g,b]], ...] → цвет
  if (v <= stops[0][0]) return stops[0][1];
  for (let k = 1; k < stops.length; k++) if (v <= stops[k][0]) {
    const [x0, c0] = stops[k - 1], [x1, c1] = stops[k], t = (v - x0) / (x1 - x0);
    return [lerp(c0[0], c1[0], t), lerp(c0[1], c1[1], t), lerp(c0[2], c1[2], t)];
  }
  return stops[stops.length - 1][1];
}
const ETA_RAMP = [[-1, [0.13, 0.35, 0.85]], [0, [0.88, 0.93, 0.98]], [1, [0.92, 0.18, 0.18]]];
const NONLIN_RAMP = [[0, [0.05, 0.20, 0.45]], [0.05, [0.10, 0.45, 0.70]], [0.1, [0.98, 0.86, 0.25]],
  [0.2, [1.0, 0.52, 0.10]], [0.3, [1.0, 0.18, 0.12]], [0.6, [1.0, 0.0, 0.35]]];

function updateWater() {
  const m = state.manifest, n = m.nx * m.ny;
  const [f, w] = frameAt(state.t);
  const s = state.scenario.eta_scale, E = state.eta;
  const o0 = f * n, o1 = (f + 1) * n;
  const { pos, nrm, col, glow, idx, wts, depth, taper, inModel } = water;
  const ex = state.exag;
  for (let q = 0; q < inModel.length; q++) {
    const v = inModel[q];
    let e = 0;
    for (let k = 0; k < 4; k++) {
      const c = idx[4 * v + k], wt = wts[4 * v + k];
      if (wt) e += wt * ((1 - w) * E[o0 + c] + w * E[o1 + c]);
    }
    e *= s * taper[v];
    pos[3 * v + 2] = SURFACE_LIFT + e * ex;
    const r = depth[v] > 0 ? Math.max(e, 0) / depth[v] : 0;
    let c = [0, 0, 0];
    if (state.mode === "eta") c = ramp(ETA_RAMP, e / state.etaPeak);
    else if (state.mode === "nonlin") c = ramp(NONLIN_RAMP, r);
    else c = [e / state.etaRef, 0, 0];                       // режим «Вода»: η для подсветки гребня
    col[3 * v] = c[0]; col[3 * v + 1] = c[1]; col[3 * v + 2] = c[2];
    glow[v] = Math.min(Math.max((r - 0.1) / 0.2, 0), 1);
  }
  // Нормали длинной волны — разностями по сетке.
  for (let q = 0; q < inModel.length; q++) {
    const v = inModel[q], i = v % NXV, j = (v - i) / NXV;
    const l = j * NXV + Math.max(i - 1, 0), r = j * NXV + Math.min(i + 1, NXV - 1);
    const d = Math.max(j - 1, 0) * NXV + i, u = Math.min(j + 1, NYV - 1) * NXV + i;
    const dzx = (pos[3 * r + 2] - pos[3 * l + 2]) / (pos[3 * r] - pos[3 * l] || 1);
    const dzy = (pos[3 * u + 2] - pos[3 * d + 2]) / (pos[3 * u + 1] - pos[3 * d + 1] || 1);
    const inv = 1 / Math.hypot(dzx, dzy, 1);
    nrm[3 * v] = -dzx * inv; nrm[3 * v + 1] = -dzy * inv; nrm[3 * v + 2] = inv;
  }
  const a = water.geo.attributes;
  a.position.needsUpdate = a.normal.needsUpdate = a.color.needsUpdate = a.glow.needsUpdate = true;
  const un = water.mesh.material.uniforms;
  un.overlay.value = state.mode === "water" ? 0 : 0.8;
  un.wind.value = state.scenario.wind;
}

// Значение η сценария в произвольной точке (lon, lat) — для среза 2D и трассеров.
function sampleField(arr, offset, scale, lon, lat) {
  const m = state.manifest;
  const fx = (lon - m.lon[0]) / (m.lon[m.nx - 1] - m.lon[0]) * (m.nx - 1);
  const fy = (lat - m.lat[0]) / (m.lat[m.ny - 1] - m.lat[0]) * (m.ny - 1);
  if (fx < 0 || fy < 0 || fx > m.nx - 1 || fy > m.ny - 1) return null;
  const ix = Math.min(Math.floor(fx), m.nx - 2), iy = Math.min(Math.floor(fy), m.ny - 2);
  const a = fx - ix, b = fy - iy;
  let sum = 0, ws = 0;
  for (const [cy, cx, w] of [[iy, ix, (1 - a) * (1 - b)], [iy, ix + 1, a * (1 - b)], [iy + 1, ix, (1 - a) * b], [iy + 1, ix + 1, a * b]]) {
    if (!m.wet[cy][cx]) continue;
    sum += w * arr[offset + cy * m.nx + cx]; ws += w;
  }
  return ws > 0.05 ? sum / ws * scale : null;
}

// ---------------------------------------------------------------------------
// Частицы: ветер над водой и трассеры течений по полю (u, v)
// ---------------------------------------------------------------------------
const N_TRACERS = 2600, N_WIND = 1400;
let tracers, wind;

function bbox() {
  const m = state.manifest;
  return { lon0: m.lon[0], lon1: m.lon[m.nx - 1], lat0: m.lat[0], lat1: m.lat[m.ny - 1] };
}
function randomWet() {
  const m = state.manifest, b = bbox();
  for (let k = 0; k < 50; k++) {
    const lon = b.lon0 + Math.random() * (b.lon1 - b.lon0), lat = b.lat0 + Math.random() * (b.lat1 - b.lat0);
    const iy = Math.round((lat - b.lat0) / (b.lat1 - b.lat0) * (m.ny - 1));
    const ix = Math.round((lon - b.lon0) / (b.lon1 - b.lon0) * (m.nx - 1));
    if (m.wet[iy][ix]) return [lon, lat];
  }
  return [b.lon0 + 1, 60];
}

function buildParticles(scene) {
  const mk = (n, size, color, opacity) => {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(n * 3), 3));
    geo.setAttribute("alpha", new THREE.BufferAttribute(new Float32Array(n), 1));
    const mat = new THREE.ShaderMaterial({
      transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
      uniforms: { color: { value: new THREE.Color(color) }, size: { value: size }, opacity: { value: opacity } },
      vertexShader: `attribute float alpha; varying float vA; uniform float size;
        void main() { vA = alpha; gl_Position = projectionMatrix * vec4(position, 1.0); gl_PointSize = size; }`,
      fragmentShader: `uniform vec3 color; uniform float opacity; varying float vA;
        void main() { vec2 d = gl_PointCoord - 0.5; float r = dot(d, d);
          if (r > 0.25) discard; gl_FragColor = vec4(color, opacity * vA * (1.0 - 4.0 * r)); }`,
    });
    const pts = new THREE.Points(geo, mat);
    pts.frustumCulled = false;
    scene.add(pts);
    return { pts, geo, n, lonlat: new Float32Array(n * 2), age: new Float32Array(n), life: new Float32Array(n) };
  };
  tracers = mk(N_TRACERS, 3.0, "#bff6ff", 0.9);
  wind = mk(N_WIND, 2.2, "#ffffff", 0.55);
  for (const P of [tracers, wind]) for (let i = 0; i < P.n; i++) respawn(P, i, true);
}
function respawn(P, i, randomAge) {
  const [lon, lat] = P === wind ? [bbox().lon0 + Math.random() * (bbox().lon1 - bbox().lon0), bbox().lat0 + Math.random() * (bbox().lat1 - bbox().lat0)] : randomWet();
  P.lonlat[2 * i] = lon; P.lonlat[2 * i + 1] = lat;
  P.life[i] = 4 + Math.random() * 6;
  P.age[i] = randomAge ? Math.random() * P.life[i] : 0;
}

function updateParticles(dtReal) {
  const m = state.manifest, s = state.scenario, n = m.nx * m.ny;
  const dtModel = dtReal * state.speed;
  const [f] = frameAt(state.t);
  const fu = Math.min(Math.round(f / s.uv_every), Math.floor((s.frames - 1) / s.uv_every));
  const nUV = state.uv.length / (2 * n);
  const ou = fu * n, ov = (nUV + fu) * n;
  const degLat = 1 / 111320, degLon = degLat / Math.cos(60 * Math.PI / 180);
  const [f0, w] = frameAt(state.t), o0 = f0 * n;
  // Трассеры течений: смещение u·dt, v·dt (течения слабые — визуально ускорены ещё ×4).
  const tp = tracers.geo.attributes.position.array, ta = tracers.geo.attributes.alpha.array;
  for (let i = 0; i < tracers.n; i++) {
    let lon = tracers.lonlat[2 * i], lat = tracers.lonlat[2 * i + 1];
    const u = sampleField(state.uv, ou, s.uv_scale, lon, lat), v = sampleField(state.uv, ov, s.uv_scale, lon, lat);
    tracers.age[i] += dtReal;
    if (u === null || v === null || tracers.age[i] > tracers.life[i]) { respawn(tracers, i, false); continue; }
    lon += u * dtModel * 4 * degLon; lat += v * dtModel * 4 * degLat;
    tracers.lonlat[2 * i] = lon; tracers.lonlat[2 * i + 1] = lat;
    const e = sampleField(state.eta, o0, s.eta_scale, lon, lat) || 0;
    const [x, y] = local(lon, lat);
    tp[3 * i] = x; tp[3 * i + 1] = y; tp[3 * i + 2] = SURFACE_LIFT + e * state.exag + 40;
    const sp = Math.hypot(u, v);
    const life = tracers.age[i] / tracers.life[i];
    ta[i] = Math.min(1, 0.25 + sp * 2.5) * Math.sin(Math.PI * life);
  }
  tracers.geo.attributes.position.needsUpdate = tracers.geo.attributes.alpha.needsUpdate = true;
  // Ветер: штрихи летят на восток со скоростью, пропорциональной ветру сценария.
  const wp = wind.geo.attributes.position.array, wa = wind.geo.attributes.alpha.array;
  const ws = s.wind, b = bbox();
  wind.pts.visible = ws > 0;
  if (ws > 0) for (let i = 0; i < wind.n; i++) {
    wind.age[i] += dtReal;
    const lon = wind.lonlat[2 * i] + ws * dtReal * 260 * degLon;
    if (lon > b.lon1 || wind.age[i] > wind.life[i]) { respawn(wind, i, false); continue; }
    wind.lonlat[2 * i] = lon;
    const [x, y] = local(lon, wind.lonlat[2 * i + 1]);
    wp[3 * i] = x; wp[3 * i + 1] = y; wp[3 * i + 2] = 900 + (i % 7) * 120;
    wa[i] = Math.sin(Math.PI * wind.age[i] / wind.life[i]) * Math.min(1, ws / 25);
  }
  wind.geo.attributes.position.needsUpdate = wind.geo.attributes.alpha.needsUpdate = true;
}

// ---------------------------------------------------------------------------
// Разрез: «кулиса» вдоль пути волны
// ---------------------------------------------------------------------------
let scene3;
// Кулиса — голографический «забор» над водой вдоль пути волны: снизу дно (пропорционально
// глубине), выше толща воды, сверху линии воды трёх моделей с собственным масштабом.
const CURTAIN = { base: 700, depth: 2600, wave: 1500, half: 45 };   // м

function buildCurtain(sec, faded) {
  const g = new THREE.Group();
  const n = sec.s.length;
  const xy = sec.lon.map((lo, i) => local(lo, sec.lat[i]));
  const hMax = Math.max(...sec.h, 1);
  const top = CURTAIN.base + CURTAIN.depth;                 // уровень покоя на кулисе
  const etaTop = Math.max(0.2, ...MODELS.map(m => Math.max(...sec.eta[m.id].map(r => Math.max(...r.map(Math.abs))))));
  // Три вершины на узел: основание, дно, уровень покоя. Полосы: грунт и вода.
  const pos = new Float32Array(n * 3 * 3), col = new Float32Array(n * 3 * 4);
  for (let i = 0; i < n; i++) {
    const [x, y] = xy[i];
    const bed = top - CURTAIN.depth * sec.h[i] / hMax;
    pos.set([x, y, CURTAIN.base, x, y, bed, x, y, top], 9 * i);
    col.set([0.30, 0.22, 0.12, 0.85, 0.62, 0.48, 0.28, 0.9, 0.10, 0.45, 0.75, 0.35], 12 * i);
  }
  const ind = [];
  for (let i = 0; i < n - 1; i++) for (const k of [0, 1]) {
    const a = 3 * i + k, b = 3 * (i + 1) + k;
    ind.push(a, b, a + 1, a + 1, b, b + 1);
  }
  const wg = new THREE.BufferGeometry();
  wg.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  wg.setAttribute("color", new THREE.BufferAttribute(col, 4));
  wg.setIndex(ind);
  const wm = new THREE.Mesh(wg, new THREE.MeshBasicMaterial({ vertexColors: true, transparent: true,
    side: THREE.DoubleSide, depthWrite: false, opacity: faded ? 0.3 : 0.9 }));
  wm.frustumCulled = false;
  wm.renderOrder = 5;
  g.add(wm);
  // Светящаяся линия пути на воде.
  const lineGeo = new THREE.BufferGeometry().setFromPoints(xy.map(([x, y]) => new THREE.Vector3(x, y, 60)));
  const line = new THREE.Line(lineGeo, new THREE.LineBasicMaterial({ color: "#ffb020", transparent: true, opacity: faded ? 0.3 : 0.9 }));
  line.frustumCulled = false;
  g.add(line);
  // Ленты воды моделей.
  const ribbonInd = [];
  for (let i = 0; i < n - 1; i++) ribbonInd.push(2 * i, 2 * i + 2, 2 * i + 1, 2 * i + 1, 2 * i + 2, 2 * i + 3);
  const ribbons = {};
  for (const mdl of MODELS) {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(n * 2 * 3), 3));
    geo.setIndex(ribbonInd);
    const mesh = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ color: mdl.color, side: THREE.DoubleSide,
      transparent: true, opacity: faded ? 0.3 : 1, depthWrite: false }));
    mesh.frustumCulled = false;
    mesh.renderOrder = 10;
    g.add(mesh);
    ribbons[mdl.id] = geo;
  }
  // Маяк в точке щелчка.
  const [cx, cy] = xy[sec.click];
  const beam = new THREE.Mesh(new THREE.CylinderGeometry(140, 140, top + CURTAIN.wave * 1.4, 20, 1, true),
    new THREE.MeshBasicMaterial({ color: "#ffb020", transparent: true, opacity: faded ? 0.12 : 0.5,
      blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide }));
  beam.rotation.x = Math.PI / 2;
  beam.position.set(cx, cy, (top + CURTAIN.wave * 1.4) / 2);
  beam.frustumCulled = false;
  g.add(beam);
  scene3.add(g);
  return { group: g, ribbons, xy, top, k: CURTAIN.wave / etaTop };
}

function updateCurtain(obj, sec, ti) {
  for (const mdl of MODELS) {
    const p = obj.ribbons[mdl.id].attributes.position.array;
    const row = sec.eta[mdl.id][ti];
    for (let i = 0; i < row.length; i++) {
      const [x, y] = obj.xy[i];
      const wet = mdl.id !== "boussinesq" || sec.s[i] <= sec.wet_boussinesq;
      const z = obj.top + row[i] * obj.k;
      const half = wet ? CURTAIN.half * (mdl.id === "linear" ? 0.7 : 1) : 0;
      p[6 * i] = x; p[6 * i + 1] = y; p[6 * i + 2] = z - half;
      p[6 * i + 3] = x; p[6 * i + 4] = y; p[6 * i + 5] = z + half;
    }
    obj.ribbons[mdl.id].attributes.position.needsUpdate = true;
  }
}

async function openSection(lon, lat) {
  const a = +$("sec-a").value, wl = +$("sec-l").value * 1000;
  toast(`<span class="spin"></span>Считаю три модели вдоль пути волны к точке ${lat.toFixed(2)}° с. ш., ${lon.toFixed(2)}° в. д.…`);
  const t0 = performance.now();
  let sec;
  try {
    const r = await fetch("/section", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ lon, lat, a, wavelength: wl }) });
    if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
    sec = await r.json();
  } catch (e) { toast(String(e.message || e), true); return; }
  toast(`Готово за ${((performance.now() - t0) / 1000).toFixed(1)} с`, false, 1500);
  sec.query = { lon, lat };
  if (state.curtain) {
    scene3.remove(state.prevCurtain?.group);
    state.prevCurtain = state.curtain;
    state.prevCurtain.group.traverse(o => { if (o.material) o.material.opacity *= 0.3; });
    state.prevSection = state.section;
  }
  state.section = sec;
  state.curtain = buildCurtain(sec, false);
  state.secT = 0; state.secPlaying = true;
  $("sec-play").textContent = "❚❚";
  $("section").classList.add("on");
  fillSectionTable(sec);
  // Камера — сбоку от пути, смотрит на точку щелчка.
  const k = sec.click, k0 = Math.max(0, k - 25);
  const dLon = sec.lon[k] - sec.lon[k0], dLat = sec.lat[k] - sec.lat[k0];
  const along = Math.atan2(dLon * Math.cos(60 * Math.PI / 180), dLat) * 180 / Math.PI;
  map.flyTo({ center: [sec.lon[k], sec.lat[k]], zoom: Math.max(map.getZoom(), 9.3), pitch: 70,
    bearing: along - 70, duration: 2200, essential: true });
}

function fillSectionTable(sec) {
  const sm = sec.summary;
  $("sec-title").textContent = `${(sm.distance / 1000).toFixed(0)} км пути · глубина ${sm.depth.toFixed(1)} м`;
  const rows = MODELS.map(mdl => {
    const e = sm.max_eta[mdl.id];
    return `<tr><td style="color:${mdl.color}">${mdl.name}</td><td>${e.toFixed(2)} м</td><td>${(e / Math.max(sm.depth, 1e-6)).toFixed(3)}</td><td>${(e / sm.a).toFixed(2)}×</td></tr>`;
  }).join("");
  $("sec-table").innerHTML = `<tr><th>в точке</th><th>η_max</th><th>a/h</th><th>к входной a</th></tr>${rows}`;
  const lin = sm.max_eta.linear, sv = sm.max_eta.saint_venant, b = sm.max_eta.boussinesq;
  const notes = [];
  if (sm.steep2_distance !== null) notes.push(`Фронт у Сен-Венана вдвое круче линейного уже через <b>${(sm.steep2_distance / 1000).toFixed(0)} км</b> пути — накопленная нелинейность.`);
  if (lin > 1.4 * Math.max(sv, b)) notes.push(`Линейная теория завышает волну здесь в <b>${(lin / Math.max(sv, b)).toFixed(1)} раза</b>: бор и обрушение она не описывает.`);
  if (b > 1.3 * sv) notes.push(`Буссинеск выше Сен-Венана: дисперсия превратила фронт в цепочку волн, а не в бор.`);
  if (lin / sm.depth > 0.1) notes.push(`a/h ≈ ${(lin / sm.depth).toFixed(2)} — включилась и <b>местная</b> нелинейность.`);
  if (!notes.length) notes.push("Здесь модели почти совпадают: волна ещё линейна.");
  notes.push(`Волна приходит через ${(sm.arrival / 60).toFixed(0)} мин, до берега дальше ${(sm.to_shore / 1000).toFixed(0)} км. Расчёт ${sm.elapsed.toFixed(1)} с.`);
  $("sec-note").innerHTML = notes.join("<br>");
}

function drawSection() {
  const sec = state.section, cv = $("sec-canvas"), ctx = cv.getContext("2d");
  const W = cv.width, Hh = cv.height, pad = 34;
  ctx.clearRect(0, 0, W, Hh);
  if (!sec) return;
  const ti = Math.min(sec.t.length - 1, Math.floor(state.secT));
  const sMax = sec.s[sec.s.length - 1];
  const X = s => pad + (W - pad - 8) * s / sMax;
  const etaTop = Math.max(0.3, ...MODELS.map(m => Math.max(...sec.eta[m.id].map(r => Math.max(...r))))) * 1.15;
  const hMax = Math.max(...sec.h);
  const yEta = e => 18 + (1 - (e + etaTop * 0.6) / (etaTop * 1.6)) * (Hh * 0.55);
  const y0 = yEta(0), yBed = h => y0 + (Hh - y0 - 16) * h / hMax;
  // Дно.
  ctx.beginPath(); ctx.moveTo(X(0), Hh);
  sec.s.forEach((s, i) => ctx.lineTo(X(s), yBed(sec.h[i])));
  ctx.lineTo(X(sMax), Hh); ctx.closePath();
  ctx.fillStyle = "rgba(140,105,60,.55)"; ctx.fill();
  ctx.fillStyle = "rgba(20,70,120,.25)";
  ctx.beginPath(); ctx.moveTo(X(0), y0);
  sec.s.forEach((s, i) => ctx.lineTo(X(s), yBed(sec.h[i])));
  ctx.lineTo(X(sMax), y0); ctx.fill();
  // Оси.
  ctx.strokeStyle = "rgba(255,255,255,.15)"; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(pad, y0); ctx.lineTo(W - 8, y0); ctx.stroke();
  ctx.fillStyle = "#8fa6bd"; ctx.font = "11px Inter";
  for (let km = 0; km <= sMax / 1000; km += 50) ctx.fillText(`${km}`, X(km * 1000) - 6, Hh - 3);
  ctx.fillText("км", W - 22, Hh - 3);
  ctx.fillText(`+${(etaTop * 0.9).toFixed(1)} м`, 0, yEta(etaTop * 0.9) + 4);
  ctx.fillText(`${hMax.toFixed(0)} м`, 2, Hh - 18);
  // Срез текущего 2D-сценария по тому же пути (фиолетовый).
  const m = state.manifest, n = m.nx * m.ny, [f] = frameAt(state.t);
  ctx.strokeStyle = "#8f7bff"; ctx.lineWidth = 1.2; ctx.setLineDash([4, 3]); ctx.beginPath();
  let started = false;
  sec.s.forEach((s, i) => {
    const e = sampleField(state.eta, f * n, state.scenario.eta_scale, sec.lon[i], sec.lat[i]);
    if (e === null) return;
    const yy = yEta(e);
    started ? ctx.lineTo(X(s), yy) : ctx.moveTo(X(s), yy); started = true;
  });
  ctx.stroke(); ctx.setLineDash([]);
  // Модели.
  for (const mdl of MODELS) {
    ctx.strokeStyle = mdl.color; ctx.lineWidth = mdl.id === "linear" ? 1.4 : 2;
    ctx.beginPath();
    sec.eta[mdl.id][ti].forEach((e, i) => {
      if (mdl.id === "boussinesq" && sec.s[i] > sec.wet_boussinesq) return;
      i ? ctx.lineTo(X(sec.s[i]), yEta(e)) : ctx.moveTo(X(sec.s[i]), yEta(e));
    });
    ctx.stroke();
  }
  // Точка щелчка.
  const xc = X(sec.s[sec.click]);
  ctx.strokeStyle = "#ffb020"; ctx.setLineDash([2, 3]);
  ctx.beginPath(); ctx.moveTo(xc, 10); ctx.lineTo(xc, Hh - 14); ctx.stroke(); ctx.setLineDash([]);
  ctx.fillStyle = "#ffb020"; ctx.fillText("точка", xc + 4, 14);
  const sm = sec.summary;
  if (sm.steep2_distance !== null) {
    const xs = X(sm.steep2_distance);
    ctx.fillStyle = "#ff7a2f"; ctx.beginPath(); ctx.arc(xs, y0, 4, 0, 7); ctx.fill();
    ctx.fillText("фронт ×2", xs + 6, y0 - 6);
  }
  $("sec-t").textContent = fmtClock(sec.t[ti]);
  $("sec-scrub").value = 1000 * ti / (sec.t.length - 1);
}

// ---------------------------------------------------------------------------
// Уровнемер у Петербурга
// ---------------------------------------------------------------------------
let gaugeBase;
function drawGaugeBase() {
  const cv = $("gauge-canvas"), ctx = cv.getContext("2d");
  const s = state.scenario, m = state.manifest;
  const W = cv.width, H = cv.height, padL = 30, padB = 16;
  const top = Math.max(4.5, s.spb_max * 1.1);
  const X = t => padL + (W - padL - 6) * t / s.t[s.t.length - 1];
  const Y = v => 6 + (H - padB - 6) * (1 - (v + 0.5) / (top + 0.5));
  ctx.clearRect(0, 0, W, H);
  ctx.font = "10px Inter";
  for (const fl of m.floods) {
    ctx.strokeStyle = "rgba(255,80,90,.45)"; ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(padL, Y(fl.level)); ctx.lineTo(W - 6, Y(fl.level)); ctx.stroke();
    ctx.setLineDash([]); ctx.fillStyle = "rgba(255,120,130,.85)";
    ctx.fillText(`${fl.year} · ${fl.level.toFixed(2)}`, W - 78, Y(fl.level) - 3);
  }
  ctx.fillStyle = "#8fa6bd";
  for (let v = 0; v <= top; v += 1) ctx.fillText(`${v}`, 8, Y(v) + 3);
  const hrs = s.t[s.t.length - 1] / 3600, stepH = hrs > 8 ? 3 : 1;
  for (let hr = 0; hr <= hrs + 1e-6; hr += stepH) ctx.fillText(`${hr} ч`, X(hr * 3600) - 6, H - 3);
  ctx.strokeStyle = "#ffb020"; ctx.lineWidth = 2; ctx.beginPath();
  s.spb_level.forEach((v, i) => i ? ctx.lineTo(X(s.t[i]), Y(v)) : ctx.moveTo(X(s.t[i]), Y(v)));
  ctx.stroke();
  gaugeBase = { img: ctx.getImageData(0, 0, W, H), X, Y };
}
function drawGauge() {
  const cv = $("gauge-canvas"), ctx = cv.getContext("2d"), m = state.manifest, s = state.scenario;
  ctx.putImageData(gaugeBase.img, 0, 0);
  const [f, w] = frameAt(state.t);
  const lv = lerp(s.spb_level[f], s.spb_level[f + 1] ?? s.spb_level[f], w);
  const x = gaugeBase.X(state.t), y = gaugeBase.Y(lv);
  ctx.strokeStyle = "rgba(255,255,255,.4)"; ctx.beginPath(); ctx.moveTo(x, 4); ctx.lineTo(x, cv.height - 14); ctx.stroke();
  ctx.fillStyle = "#fff"; ctx.beginPath(); ctx.arc(x, y, 4, 0, 7); ctx.fill();
  $("gauge-now").textContent = `${lv >= 0 ? "+" : ""}${lv.toFixed(2)} м`;
  const passed = m.floods.filter(fl => lv >= fl.level - 0.005).sort((a, b) => b.level - a.level)[0];
  $("alarm").textContent = passed ? `УРОВЕНЬ +${lv.toFixed(2)} м — КАК В ${passed.year} ГОДУ` : lv > 1.6 ? "ОПАСНЫЙ ПОДЪЁМ ВОДЫ" : "";
  $("alarm").classList.toggle("on", !!passed);
}

// ---------------------------------------------------------------------------
// Интерфейс: время, фокусы, полёт, тосты
// ---------------------------------------------------------------------------
const fmtClock = t => `${Math.floor(t / 3600)}:${String(Math.floor(t / 60) % 60).padStart(2, "0")}`;
let toastTimer;
function toast(html, err = false, ms = 0) {
  const el = $("toast");
  el.innerHTML = html; el.classList.add("on"); el.classList.toggle("err", err);
  clearTimeout(toastTimer);
  if (ms || err) toastTimer = setTimeout(() => el.classList.remove("on"), ms || 4000);
}
function banner(kicker, title) {
  $("banner-kicker").textContent = kicker; $("banner-title").textContent = title;
  $("banner").classList.add("on"); document.body.classList.add("cinema");
  setTimeout(() => { $("banner").classList.remove("on"); document.body.classList.remove("cinema"); }, 2600);
}
function goFocus(k) {
  const f = FOCI[k];
  document.querySelectorAll("#foci button").forEach((b, i) => b.classList.toggle("on", i === k));
  banner(f.kicker, f.title);
  map.flyTo({ center: f.center, zoom: f.zoom, pitch: f.pitch, bearing: f.bearing, duration: 3200, curve: 1.6, essential: true });
}

const keys = new Set();
function fly(dt) {
  if (!keys.size) return;
  const fast = keys.has("shift") ? 3 : 1, v = 650 * dt * fast;
  let dx = 0, dy = 0;
  if (keys.has("w")) dy -= v; if (keys.has("s")) dy += v;
  if (keys.has("a")) dx -= v; if (keys.has("d")) dx += v;
  if (dx || dy) map.panBy([dx, dy], { duration: 0 });
  if (keys.has("q")) map.setBearing(map.getBearing() - 50 * dt * fast);
  if (keys.has("e")) map.setBearing(map.getBearing() + 50 * dt * fast);
  if (keys.has("r")) map.setPitch(Math.min(map.getPitch() + 30 * dt, 85));
  if (keys.has("f")) map.setPitch(Math.max(map.getPitch() - 30 * dt, 0));
  if (keys.has("z")) map.setZoom(map.getZoom() + 1.2 * dt * fast);
  if (keys.has("x")) map.setZoom(map.getZoom() - 1.2 * dt * fast);
}
const RU = { "ц": "w", "ф": "a", "ы": "s", "в": "d", "й": "q", "у": "e", "к": "r", "а": "f", "я": "z", "ч": "x", "р": "h" };
function keyName(e) { const k = e.key.toLowerCase(); return e.key === "Shift" ? "shift" : RU[k] || k; }
addEventListener("keydown", e => {
  if (e.target.closest?.("input, select")) return;
  const k = keyName(e);
  if (["1", "2", "3", "4", "5", "6"].includes(k)) goFocus(+k - 1);
  else if (k === " ") { e.preventDefault(); togglePlay(); }
  else if (k === "h") $("hud").classList.toggle("hidden");
  else if (k === "escape") closeSection();
  else keys.add(k);
});
addEventListener("keyup", e => keys.delete(keyName(e)));
addEventListener("blur", () => keys.clear());

function togglePlay() { state.playing = !state.playing; $("play").textContent = state.playing ? "❚❚" : "▶"; }
function closeSection() {
  $("section").classList.remove("on");
  for (const c of [state.curtain, state.prevCurtain]) if (c) scene3.remove(c.group);
  state.curtain = state.prevCurtain = state.section = null;
}

// ---------------------------------------------------------------------------
// Карта и слой three.js
// ---------------------------------------------------------------------------
// Свой стиль без шрифтов и спрайтов из сети: страница грузится сразу, тайлы идут
// через кэш сервера (showcase/tiles.py), подписи важных мест — свои метки.
const STYLE = {
  version: 8,
  sources: {
    sat: { type: "raster", tiles: ["/tiles/sat/{z}/{x}/{y}"], tileSize: 256, maxzoom: 10,
      attribution: "Снимок и подписи: Esri, Maxar, Earthstar Geographics" },
    labels: { type: "raster", tiles: ["/tiles/labels/{z}/{x}/{y}"], tileSize: 256, maxzoom: 10 },
  },
  layers: [
    { id: "bg", type: "background", paint: { "background-color": "#2a3526" } },
    { id: "sat", type: "raster", source: "sat", paint: { "raster-saturation": -0.15, "raster-contrast": 0.1, "raster-brightness-max": 0.92 } },
    { id: "labels", type: "raster", source: "labels", maxzoom: 12.5, paint: { "raster-opacity": 0.85 } },
  ],
};
const map = new maplibregl.Map({
  container: "map",
  style: STYLE,
  center: FOCI[0].center, zoom: 5.6, pitch: 30, bearing: 0, maxPitch: 85,
  attributionControl: { compact: true }, antialias: true,
});
window.map = map;
window.gulf = state;                 // для отладки из консоли
window.gulfPerf = () => {            // время одного кадра без отрисовки, мс
  const t0 = performance.now(); updateWater(); const t1 = performance.now();
  updateParticles(1 / 60); const t2 = performance.now(); drawGauge(); const t3 = performance.now();
  let zmax = 0; for (const v of water.inModel) zmax = Math.max(zmax, water.pos[3 * v + 2]);
  return { water: t1 - t0, particles: t2 - t1, gauge: t3 - t2, inModel: water.inModel.length, zmax,
    overlay: water.mesh.material.uniforms.overlay.value, octaves: water.mesh.material.uniforms.octaves.value };
};

const waterLayer = {
  id: "gulf-3d", type: "custom", renderingMode: "3d",
  onAdd(mp, gl) {
    this.camera = new THREE.Camera();
    scene3 = this.scene = new THREE.Scene();
    this.renderer = new THREE.WebGLRenderer({ canvas: mp.getCanvas(), context: gl, antialias: true });
    this.renderer.autoClear = false;
    buildWater(this.scene);
    buildParticles(this.scene);
    city = new City(this.renderer);
    window.gulfCity = city;
  },
  prerender() {
    // Шаги GPU-затопления до основной отрисовки кадра.
    if (!cityState.area) return;
    this.renderer.resetState();
    city.simulate(cityState.area, cityState.level, cityState.running);
    this.renderer.resetState();
  },
  render(gl, matrix) {
    const M = new THREE.Matrix4().fromArray(matrix).multiply(
      new THREE.Matrix4().makeTranslation(origin.x, origin.y, origin.z)
        .scale(new THREE.Vector3(meter, -meter, meter)));
    this.camera.projectionMatrix = M;
    const inv = M.clone().invert();
    const c = new THREE.Vector4(0, 0, 1, 0).applyMatrix4(inv);
    water.mesh.material.uniforms.camPos.value.set(c.x / c.w, c.y / c.w, c.z / c.w);
    water.mesh.material.uniforms.time.value = performance.now() / 1000;
    this.renderer.resetState();
    this.renderer.render(this.scene, this.camera);
    if (cityState.area) city.render(cityState.area, matrix);
  },
};

map.on("load", async () => {
  const m = await loadManifest();
  await loadSea();
  origin = maplibregl.MercatorCoordinate.fromLngLat([m.lon[0], m.lat[0]], 0);
  meter = origin.meterInMercatorCoordinateUnits();

  map.addSource("dem", { type: "raster-dem", encoding: "terrarium", tileSize: 256, maxzoom: 9,
    tiles: ["/tiles/dem/{z}/{x}/{y}"], attribution: "Рельеф: Mapzen, AWS Terrain Tiles" });
  const firstSymbol = "labels";
  map.setTerrain({ source: "dem", exaggeration: TERRAIN_EXAG });
  if (map.setSky) map.setSky({ "sky-color": "#6fa8dc", "horizon-color": "#dce9f5", "fog-color": "#c9d8e6",
    "sky-horizon-blend": 0.6, "horizon-fog-blend": 0.6, "fog-ground-blend": 0.4 });

  // Рамка расчётной области и уровнемер.
  const b = bbox();
  map.addSource("domain", { type: "geojson", data: { type: "Feature", geometry: { type: "LineString",
    coordinates: [[b.lon0, b.lat0], [b.lon1, b.lat0], [b.lon1, b.lat1], [b.lon0, b.lat1], [b.lon0, b.lat0]] } } });
  map.addLayer({ id: "domain", type: "line", source: "domain",
    paint: { "line-color": "#ffb020", "line-width": 1.6, "line-dasharray": [3, 2], "line-opacity": 0.8 } });
  for (const p of [{ name: "Уровнемер · Горный институт", lon: m.gauge.lon, lat: m.gauge.lat },
    ...m.places.filter(p => p.name !== "Санкт-Петербург"),
    { name: "Санкт-Петербург", lon: 30.32, lat: 59.94 }, { name: "Хельсинки", lon: 24.94, lat: 60.17 },
    { name: "Таллин", lon: 24.75, lat: 59.44 }, { name: "Котка", lon: 26.94, lat: 60.47 }]) {
    const el = document.createElement("div");
    el.className = "pin";
    el.innerHTML = `<i></i><span>${p.name}</span>`;
    new maplibregl.Marker({ element: el, anchor: "bottom" }).setLngLat([p.lon, p.lat]).addTo(map);
  }

  await loadScenario($("scenario").value);
  map.addLayer(waterLayer, firstSymbol);
  setTimeout(() => goFocus(0), 400);
  requestAnimationFrame(loop);
});

map.on("click", e => {
  const b = bbox(), { lng, lat } = e.lngLat;
  if (lng < b.lon0 || lng > b.lon1 || lat < b.lat0 || lat > b.lat1) {
    toast("Модель считает восточную часть залива (оранжевая рамка, 27–30,4° в. д.) — щёлкните внутри неё", true);
    return;
  }
  openSection(lng, lat);
});

// ---------------------------------------------------------------------------
// Детальный город: включается при приближении к зоне
// ---------------------------------------------------------------------------
function areaAt(lon, lat) {
  return CITY_AREAS.find(a => lat > a.bbox[0] && lat < a.bbox[2] && lon > a.bbox[1] && lon < a.bbox[3]);
}
// Уровень моря у зоны из текущего 2D-сценария (ячейка у края зоны, см. flood_map.GAUGE).
function seaLevelAt(meta) {
  const m = state.manifest, n = m.nx * m.ny, [f, w] = frameAt(state.t);
  const c = meta.gauge.iy * m.nx + meta.gauge.ix, s = state.scenario.eta_scale;
  return s * ((1 - w) * state.eta[f * n + c] + w * state.eta[(f + 1) * n + c]);
}
async function enterCity(desc) {
  cityState.loading = true;
  toast(`<span class="spin"></span>Строю ${desc.name}: здания, улицы, набережные…`);
  let area;
  try { area = await city.load(desc.id); } catch (e) { toast(`Нет данных города: ${e.message || e}`, true); cityState.loading = false; return; }
  cityState.loading = false;
  if (cityState.area) return;
  cityState.area = area;
  cityState.saved = { exag: state.exag, terrain: map.getTerrain() };
  state.exag = 1;
  $("exag").value = 1; $("exag-v").textContent = "1";
  map.setTerrain(null);                                  // в городе все высоты настоящие, в метрах
  const [x0, y0] = local(desc.bbox[1], desc.bbox[0]), [x1, y1] = local(desc.bbox[3], desc.bbox[2]);
  water.mesh.material.uniforms.cityBox.value.set(x0, y0, x1, y1);
  $("city").classList.add("on");
  $("city-name").textContent = desc.name;
  $("city-stats").textContent = `${area.nBuildings.toLocaleString("ru")} зданий OSM в ${area.chunks.length} кварталах · ` +
    `${Math.round(area.nVerts / 1000)} тыс. вершин · затопление ${area.meta.nx} × ${area.meta.ny} по ${area.meta.res.toFixed(1)} м · сборка ${(area.buildMs / 1000).toFixed(1)} с`;
  water.geo.setIndex(water.trisCoarse);                  // вода залива в городе — вдвое реже (вершин ≤ 150 тыс.)
  toast(`${desc.name}: ${area.nBuildings.toLocaleString("ru")} зданий`, false, 2500);
}
function leaveCity() {
  const area = cityState.area;
  if (!area) return;
  cityState.area = null;
  state.exag = cityState.saved.exag;
  $("exag").value = state.exag; $("exag-v").textContent = state.exag;
  map.setTerrain(cityState.saved.terrain);
  water.mesh.material.uniforms.cityBox.value.set(1e9, 1e9, -1e9, -1e9);
  water.geo.setIndex(water.trisFull);
  $("city").classList.remove("on");
}
function updateCity(dt) {
  if (!city || !state.manifest) return;
  const c = map.getCenter(), z = map.getZoom();
  const desc = z >= CITY_ZOOM ? areaAt(c.lng, c.lat) : null;
  if (cityState.area && (!desc || desc.id !== cityState.area.id)) leaveCity();
  if (desc && !cityState.area && !cityState.loading) enterCity(desc);
  if (!cityState.area) { cityState.running = false; return; }
  cityState.level = cityState.manual ? cityState.manualLevel : seaLevelAt(cityState.area.meta);
  cityState.running = state.playing;                    // частоту шагов ограничивает city.simulate
  $("city-level").textContent = `${cityState.level >= 0 ? "+" : ""}${cityState.level.toFixed(2)} м`;
}

// ---------------------------------------------------------------------------
// Главный цикл
// ---------------------------------------------------------------------------
let last = performance.now(), fpsAcc = 0, fpsN = 0;
function loop(now) {
  const dt = Math.min((now - last) / 1000, 0.1);
  last = now;
  fpsAcc += dt; fpsN++;
  if (fpsAcc > 0.5) {
    const fps = fpsN / fpsAcc;
    $("fps").textContent = `${Math.round(fps)} fps`;
    // Автонастройка: слабой видеокарте — меньше октав ряби и меньше пикселей.
    const oct = water.mesh.material.uniforms.octaves;
    if (document.visibilityState === "visible" && fps > 3) {
      if (fps < 28 && oct.value > 5) oct.value -= 2;
      else if (fps < 22 && map.getPixelRatio() > 1) map.setPixelRatio(1);
      else if (fps > 55 && oct.value < 11) oct.value += 1;
    }
    fpsAcc = fpsN = 0;
  }
  const sc = state.scenario, tEnd = sc.t[sc.t.length - 1];
  if (state.playing) state.t = (state.t + dt * state.speed) % tEnd;
  $("time").textContent = fmtClock(state.t);
  $("scrub").value = 1000 * state.t / tEnd;
  fly(dt);
  updateCity(dt);
  updateWater();
  // Частицы ветра и течений в масштабе города не видны — не тратим на них кадр.
  tracers.pts.visible = wind.pts.visible = !cityState.area;
  if (!cityState.area) updateParticles(state.playing ? dt : 0);
  drawGauge();
  if (state.section) {
    const sec = state.section;
    if (state.secPlaying) state.secT = (state.secT + dt * sec.t.length / 14) % (sec.t.length - 1);
    const ti = Math.floor(state.secT);
    updateCurtain(state.curtain, sec, ti);
    if (state.prevCurtain) updateCurtain(state.prevCurtain, state.prevSection, Math.min(ti, state.prevSection.t.length - 1));
    drawSection();
  }
  map.triggerRepaint();
  requestAnimationFrame(loop);
}

// Элементы управления.
$("scenario").addEventListener("change", async e => { await loadScenario(e.target.value); state.t = 0; });
$("play").addEventListener("click", togglePlay);
$("scrub").addEventListener("input", e => { const sc = state.scenario; if (!sc) return; state.t = e.target.value / 1000 * sc.t[sc.t.length - 1]; });
$("speed").addEventListener("change", e => { state.speed = +e.target.value; });
$("exag").addEventListener("input", e => { state.exag = +e.target.value; $("exag-v").textContent = e.target.value; });
document.querySelectorAll("#color-mode button").forEach(b => b.addEventListener("click", () => {
  state.mode = b.dataset.mode;
  document.querySelectorAll("#color-mode button").forEach(x => x.classList.toggle("on", x === b));
}));
document.querySelectorAll("#foci button").forEach(b => b.addEventListener("click", () => goFocus(+b.dataset.focus)));
$("sec-close").addEventListener("click", closeSection);
$("city-manual").addEventListener("change", e => { cityState.manual = e.target.checked; });
$("city-slider").addEventListener("input", e => { cityState.manualLevel = +e.target.value; $("city-slider-v").textContent = (+e.target.value).toFixed(2); });
$("city-reset").addEventListener("click", () => { if (cityState.area) city.reset(cityState.area); });
$("sec-play").addEventListener("click", () => { state.secPlaying = !state.secPlaying; $("sec-play").textContent = state.secPlaying ? "❚❚" : "▶"; });
$("sec-scrub").addEventListener("input", e => { if (state.section) state.secT = e.target.value / 1000 * (state.section.t.length - 1); });
$("sec-rerun").addEventListener("click", () => { const q = state.section?.query; if (q) openSection(q.lon, q.lat); });
