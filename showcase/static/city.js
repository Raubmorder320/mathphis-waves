// Детальный режим города: здания OSM с процедурными фасадами, улицы с бордюрами,
// гранитные набережные и затопление улиц на видеокарте с обтеканием каждого дома.
import * as THREE from "three";

const SUN = new THREE.Vector3(0.8, 0.12, 0.26).normalize();
const PALETTE = [            // петербургские фасады: охра, беж, фисташка, розовый, голубой, серый
  [0.86, 0.70, 0.42], [0.90, 0.82, 0.64], [0.72, 0.80, 0.62], [0.86, 0.64, 0.58],
  [0.62, 0.74, 0.82], [0.78, 0.76, 0.72], [0.93, 0.88, 0.78], [0.80, 0.56, 0.36],
];
const ROOFS = [[0.33, 0.42, 0.36], [0.42, 0.42, 0.44], [0.50, 0.30, 0.22], [0.30, 0.32, 0.34]];
const CSS_COLORS = { white: "#efefea", yellow: "#d9b45c", beige: "#d8c4a0", grey: "#9a9a98", gray: "#9a9a98",
  brown: "#8a5a3c", red: "#a8483a", pink: "#d49a90", green: "#8fa880", blue: "#8aa2bf", orange: "#d08a4a" };

function hash(n) { const x = Math.sin(n * 127.1) * 43758.5453; return x - Math.floor(x); }
function parseColour(c) {
  if (!c) return null;
  const k = String(c).toLowerCase();
  try { const col = new THREE.Color(CSS_COLORS[k] || k); return [col.r, col.g, col.b]; } catch { return null; }
}

// ---------------------------------------------------------------------------
// Шейдеры
// ---------------------------------------------------------------------------
const COMMON = `
  uniform vec3 sunDir; uniform vec3 camPos;
  float hash12(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
  float vnoise(vec2 p) { vec2 i = floor(p), f = fract(p); vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash12(i), hash12(i + vec2(1, 0)), u.x), mix(hash12(i + vec2(0, 1)), hash12(i + vec2(1, 1)), u.x), u.y); }
  vec3 skyColor(vec3 r) { return mix(vec3(0.80, 0.86, 0.92), vec3(0.24, 0.44, 0.74), pow(clamp(r.z, 0.0, 1.0), 0.6)); }
  vec3 fogged(vec3 c, vec3 p) { float f = 1.0 - exp(-length(camPos - p) / 60000.0); return mix(c, vec3(0.78, 0.84, 0.90), f * 0.8); }
`;

// Фасад: этажи и окна по метрам стены (u — вдоль стены, v — высота над землёй).
// Окно — «комната» за стеклом (interior mapping): луч зрения внутри коробки 3,2 × floor × 4 м.
const FACADE_VS = `
  attribute vec2 wuv; attribute vec4 fparam; attribute vec3 fcolor; attribute vec3 wtan;
  varying vec2 vUV; varying vec4 vParam; varying vec3 vCol; varying vec3 vN; varying vec3 vP; varying vec3 vT;
  void main() { vUV = wuv; vParam = fparam; vCol = fcolor; vN = normal; vP = position; vT = wtan;
    gl_Position = projectionMatrix * vec4(position, 1.0); }`;
const FACADE_FS = COMMON + `
  uniform float night;
  varying vec2 vUV; varying vec4 vParam; varying vec3 vCol; varying vec3 vN; varying vec3 vP; varying vec3 vT;
  void main() {
    vec3 n = normalize(vN);
    vec3 V = normalize(camPos - vP);
    float diff = max(dot(n, sunDir), 0.0);
    // Крыша.
    if (vParam.w > 1.5) {
      float seam = smoothstep(0.92, 1.0, fract(vP.x / 0.9 + vP.y * 0.0));
      vec3 c = vCol * (0.75 + 0.25 * vnoise(vP.xy / 3.0)) * (1.0 - 0.15 * seam);
      c *= 0.55 + 0.6 * max(dot(n, sunDir), 0.0);
      gl_FragColor = vec4(fogged(c, vP), 1.0); return;
    }
    float floorH = vParam.x, spacing = vParam.y, top = vParam.z;
    vec2 uv = vUV;
    float mpp = max(fwidth(uv.x), fwidth(uv.y));                     // метров на пиксель
    // Штукатурка или кирпич + грязь у земли.
    vec3 wall = vCol * (0.88 + 0.12 * vnoise(uv * vec2(1.3, 2.0)));
    if (vParam.w > 0.5) {                                              // кирпич (промзона)
      vec2 b = uv / vec2(0.25, 0.075); b.x += 0.5 * floor(b.y);
      float mortar = step(0.9, fract(b.x)) + step(0.85, fract(b.y));
      wall = mix(wall, vec3(0.62, 0.60, 0.56), clamp(mortar, 0.0, 1.0) * 0.6);
    }
    wall *= 1.0 - 0.25 * smoothstep(1.2, 0.0, uv.y);
    float plinth = step(uv.y, 0.9);
    wall = mix(wall, vec3(0.46, 0.45, 0.44) * (0.9 + 0.1 * vnoise(uv * 4.0)), plinth);
    float cornice = smoothstep(top - 0.9, top - 0.75, uv.y) * (1.0 - smoothstep(top - 0.35, top - 0.2, uv.y));
    wall = mix(wall, wall * 1.18 + 0.05, cornice);
    vec3 c = wall * (0.45 + 0.65 * diff);

    // Окна: колонка и этаж.
    float fl = floor(uv.y / floorH), fy = fract(uv.y / floorH);
    float col = floor(uv.x / spacing), fx = fract(uv.x / spacing);
    bool shop = fl < 0.5 && vParam.w < 0.5;
    vec2 lo = shop ? vec2(0.12, 0.18) : vec2(0.28, 0.30), hi = shop ? vec2(0.88, 0.86) : vec2(0.72, 0.80);
    float inWin = step(lo.x, fx) * step(fx, hi.x) * step(lo.y, fy) * step(fy, hi.y)
                * step(1.0, uv.y) * step(uv.y, top - 1.0);
    float frame = inWin * (1.0 - step(lo.x + 0.03, fx) * step(fx, hi.x - 0.03) * step(lo.y + 0.03, fy) * step(fy, hi.y - 0.03));
    float mullion = inWin * step(abs(fx - 0.5), 0.012) * (shop ? 0.0 : 1.0);
    float dist = length(camPos - vP);
    bool near = dist < 600.0 && mpp < 0.6;                         // LOD 0: окна с комнатами
    if (inWin > 0.5 && near) {
      // Interior mapping: стена с окном — плоскость, комната глубиной 4 м за ней.
      vec3 T = normalize(vT), B = vec3(0.0, 0.0, 1.0);
      vec3 d = normalize(vec3(dot(-V, T), dot(-V, B), dot(-V, -n)));   // луч в координатах комнаты
      vec2 cell = vec2((fx - 0.5) * spacing, (fy - 0.5) * floorH);
      vec3 o = vec3(cell, 0.0);
      vec3 room = vec3(spacing * 0.5, floorH * 0.5, 4.0);
      vec3 tPos = (sign(d) * room - o) / d;
      float t = min(min(tPos.x, tPos.y), tPos.z);
      vec3 hit = o + d * t;
      float seed = hash12(vec2(col, fl) + vParam.xy * 13.0);
      vec3 roomC;
      if (t == tPos.z) roomC = mix(vec3(0.55, 0.50, 0.44), vec3(0.75, 0.66, 0.52), seed);   // задняя стена
      else if (t == tPos.y) roomC = d.y > 0.0 ? vec3(0.82, 0.80, 0.76) : vec3(0.36, 0.26, 0.18); // потолок / пол
      else roomC = vec3(0.62, 0.58, 0.52);
      roomC *= 0.35 + 0.25 * (hit.z / 4.0);
      float lit = step(0.62 - 0.35 * night, seed);
      roomC = mix(roomC, vec3(1.0, 0.82, 0.52) * (0.8 + 0.4 * seed), lit * (0.25 + 0.75 * night));
      vec3 R = reflect(-V, n);
      float F = 0.04 + 0.96 * pow(1.0 - clamp(dot(n, V), 0.0, 1.0), 5.0);
      vec3 glass = mix(roomC, skyColor(R), clamp(F + 0.15, 0.0, 0.9)) + pow(max(dot(R, sunDir), 0.0), 300.0) * 2.0;
      c = glass;
    } else if (inWin > 0.5 && dist < 1500.0) {
      c = mix(c, vec3(0.22, 0.28, 0.34), 0.55);                       // издалека окна — тёмные пятна
    }
    c = mix(c, vec3(0.92, 0.90, 0.86) * (0.5 + 0.5 * diff), clamp(frame + mullion, 0.0, 1.0) * (near ? 1.0 : 0.0));
    gl_FragColor = vec4(fogged(c, vP), 1.0);
  }`;

// Дороги, тротуары, бордюры и гранит (kind: 0 асфальт, 1 тротуар, 2 бордюр/гранит, 3 земля).
const STREET_VS = `
  attribute vec2 suv; attribute float kind;
  varying vec2 vUV; varying float vKind; varying vec3 vN; varying vec3 vP;
  void main() { vUV = suv; vKind = kind; vN = normal; vP = position; gl_Position = projectionMatrix * vec4(position, 1.0); }`;
const STREET_FS = COMMON + `
  varying vec2 vUV; varying float vKind; varying vec3 vN; varying vec3 vP;
  void main() {
    vec3 n = normalize(vN);
    float diff = 0.5 + 0.6 * max(dot(n, sunDir), 0.0);
    vec3 c;
    if (vKind < 0.5) {                                                // асфальт + разметка
      c = vec3(0.20, 0.21, 0.22) * (0.85 + 0.3 * vnoise(vP.xy * 1.7)) + 0.04 * vnoise(vP.xy * 9.0);
      float lane = step(abs(vUV.y), 0.08) * step(0.5, fract(vUV.x / 6.0));   // осевая пунктиром
      c = mix(c, vec3(0.9), lane);
    } else if (vKind < 1.5) {                                         // плитка тротуара
      vec2 t = fract(vP.xy / 0.5);
      float joint = step(0.94, max(t.x, t.y));
      c = mix(vec3(0.58, 0.56, 0.53) * (0.9 + 0.2 * vnoise(floor(vP.xy / 0.5))), vec3(0.4), joint * 0.5);
    } else if (vKind < 2.5) {                                         // гранит
      c = mix(vec3(0.52, 0.42, 0.40), vec3(0.66, 0.58, 0.55), vnoise(vP.xy * 3.0 + vP.z * 5.0));
      c *= 0.9 + 0.2 * hash12(floor(vec2(vUV.x / 1.2, vP.z / 0.6)));
    } else if (vKind < 3.5) {                                         // дворы, газоны
      // Дворы и газоны: крупные пятна травы и асфальта, мелкая фактура слабее.
      float g = vnoise(vP.xy / 45.0) * 0.7 + vnoise(vP.xy / 9.0) * 0.3;
      c = mix(vec3(0.36, 0.42, 0.28), vec3(0.47, 0.46, 0.43), smoothstep(0.42, 0.58, g));
      c *= 0.94 + 0.06 * vnoise(vP.xy * 1.5);
    } else {                                                          // дно: ил и песчаная рябь
      float rip = 0.5 + 0.5 * sin(vP.x * 2.1 + 3.0 * vnoise(vP.xy / 6.0));
      c = mix(vec3(0.30, 0.27, 0.20), vec3(0.46, 0.41, 0.30), 0.6 * vnoise(vP.xy / 9.0) + 0.25 * rip);
    }
    gl_FragColor = vec4(fogged(c * diff, vP), 1.0);
  }`;

// Вода затопления: глубина и расходы из GPU-расчёта.
const FLOODW_VS = `
  uniform sampler2D state; uniform sampler2D terrain; uniform vec2 gridSize;
  attribute vec2 cuv;
  varying float vD; varying vec2 vUV; varying vec3 vP; varying float vSpeed;
  void main() {
    vec4 s = texture2D(state, cuv);
    vec4 t = texture2D(terrain, cuv);
    bool wall = t.r > 500.0;                                   // здание: вода у стены обрывается
    float bed = wall ? t.g : t.r;
    vD = wall ? 0.0 : s.r; vUV = cuv; vSpeed = length(s.gb);
    vec3 p = position;
    p.z = vD > 0.01 ? bed + vD : bed - 0.3;
    vP = p;
    gl_Position = projectionMatrix * vec4(p, 1.0);
  }`;
const FLOODW_FS = COMMON + `
  uniform float time; uniform sampler2D terrain; uniform vec2 texel;
  varying float vD; varying vec2 vUV; varying vec3 vP; varying float vSpeed;
  void main() {
    if (vD < 0.02) discard;
    vec2 p = vP.xy;
    vec2 g = vec2(0.0);
    for (int i = 0; i < 6; i++) {
      float fi = float(i), L = 14.0 * pow(0.62, fi), k = 6.2831 / L, w = sqrt(9.81 * k * tanh(k * max(vD, 0.05)));
      vec2 d = normalize(vec2(cos(fi * 2.4 + 0.3), sin(fi * 2.4 + 0.3)) + vec2(1.0, 0.1));
      g += d * 0.10 * cos(k * dot(d, p) - w * time + fi) * smoothstep(3.0, 8.0, L / max(length(fwidth(p)), 0.01));
    }
    vec3 n = normalize(vec3(-g, 1.0));
    vec3 V = normalize(camPos - vP), R = reflect(-V, n);
    float F = 0.02 + 0.98 * pow(1.0 - clamp(dot(n, V), 0.0, 1.0), 5.0);
    // Мутная вода затопления на суше, прозрачнее на мелком; пена у стен и в быстрых струях.
    vec3 body = mix(vec3(0.30, 0.34, 0.26), vec3(0.06, 0.20, 0.26), smoothstep(0.1, 2.5, vD));
    float wallNear = 0.0;
    for (int k = 0; k < 4; k++) {
      vec2 o = texel * 2.0 * vec2(k == 0 ? 1.0 : k == 1 ? -1.0 : 0.0, k == 2 ? 1.0 : k == 3 ? -1.0 : 0.0);
      wallNear += step(500.0, texture2D(terrain, vUV + o).r);
    }
    // Пена только у стен домов и в быстрых струях; издалека (крупный пиксель) гаснет.
    float mpp = max(length(fwidth(p)), 0.01);
    float foam = clamp(wallNear * 0.3 * smoothstep(3.0, 0.3, vD) + smoothstep(1.5, 3.5, vSpeed), 0.0, 1.0)
               * smoothstep(0.5, 0.8, vnoise(p * 0.35 + vec2(time * 0.25, 0.0)) * 0.7 + vnoise(p * 1.1 - time * 0.4) * 0.3)
               * smoothstep(2.5, 0.6, mpp);
    vec3 c = mix(body, skyColor(R), F) + pow(max(dot(R, sunDir), 0.0), 400.0) * 3.0;
    c = mix(c, vec3(0.92, 0.95, 0.96), foam * 0.8);
    gl_FragColor = vec4(fogged(c, vP), mix(0.55, 0.93, smoothstep(0.0, 1.0, vD)));
  }`;

// GPU-затопление: модель «виртуальных труб» (Mei et al., 2007) — упрощённая мелкая вода.
// Состояние: R — глубина d, GB — скорость (для пены). Расходы: RGBA = в лево/право/низ/верх.
const SIM_VS = `varying vec2 vUv; void main() { vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }`;
const FLUX_FS = `
  uniform sampler2D state; uniform sampler2D flux; uniform sampler2D terrain; uniform vec2 texel;
  uniform float dt; uniform float cell;
  varying vec2 vUv;
  float level(vec2 uv, out float wall) {
    vec4 t = texture2D(terrain, uv);
    wall = (uv.x < 0.0 || uv.y < 0.0 || uv.x > 1.0 || uv.y > 1.0 || t.r > 500.0) ? 1.0 : 0.0;
    return t.r + texture2D(state, uv).r;
  }
  void main() {
    float w0, wl, wr, wb, wt;
    float h0 = level(vUv, w0);
    float d = texture2D(state, vUv).r;
    vec4 f = texture2D(flux, vUv) * 0.998;                     // трение
    float hl = level(vUv - vec2(texel.x, 0.0), wl), hr = level(vUv + vec2(texel.x, 0.0), wr);
    float hb = level(vUv - vec2(0.0, texel.y), wb), ht = level(vUv + vec2(0.0, texel.y), wt);
    float k = dt * 9.81 * cell;                                 // площадь трубы cell², длина cell
    f = max(vec4(0.0), f + k * vec4(h0 - hl, h0 - hr, h0 - hb, h0 - ht));
    f *= 1.0 - vec4(wl, wr, wb, wt);
    f *= 1.0 - w0;
    float out_ = f.r + f.g + f.b + f.a;
    if (out_ > 0.0) f *= min(1.0, d * cell * cell / (out_ * dt));
    gl_FragColor = f;
  }`;
const DEPTH_FS = `
  uniform sampler2D state; uniform sampler2D flux; uniform sampler2D terrain; uniform vec2 texel;
  uniform float dt; uniform float cell; uniform float sea;
  varying vec2 vUv;
  void main() {
    vec4 t = texture2D(terrain, vUv);
    vec4 f = texture2D(flux, vUv);
    float inflow = texture2D(flux, vUv - vec2(texel.x, 0.0)).g + texture2D(flux, vUv + vec2(texel.x, 0.0)).r
                 + texture2D(flux, vUv - vec2(0.0, texel.y)).a + texture2D(flux, vUv + vec2(0.0, texel.y)).b;
    float outflow = f.r + f.g + f.b + f.a;
    float d0 = texture2D(state, vUv).r;
    float d = max(0.0, d0 + dt * (inflow - outflow) / (cell * cell));
    if (t.b > 0.5) d = max(0.0, sea - t.r);                    // край моря: уровень из 2D-расчёта
    if (t.r > 500.0) d = 0.0;                                   // в зданиях воды нет
    float dm = max(0.5 * (d0 + d), 0.05);
    vec2 vel = vec2(texture2D(flux, vUv - vec2(texel.x, 0.0)).g - f.r + f.g - texture2D(flux, vUv + vec2(texel.x, 0.0)).r,
                    texture2D(flux, vUv - vec2(0.0, texel.y)).a - f.b + f.a - texture2D(flux, vUv + vec2(0.0, texel.y)).b)
              / (2.0 * cell * dm);
    gl_FragColor = vec4(d, vel, 0.0);
  }`;

// ---------------------------------------------------------------------------
// Сборка города
// ---------------------------------------------------------------------------
const LOD_STREETS = 600;     // м — ближе видны улицы, бордюры и окна
const LOD_BUILDINGS = 3000;  // м — дальше кварталы скрыты
const SIM_HZ = 30;           // обновлений затопления в секунду, не больше
const SIM_SUBSTEPS = 6;      // шагов расчёта за обновление

export class City {
  // У каждой зоны своя сцена и свой центр координат: метры от юго-западного угла зоны.
  // Матрица «камера × центр зоны» считается в JS (двойная точность), поэтому вершины
  // не дрожат даже на бордюрах в 18 см — иначе float32 в 180 км от центра залива даёт метры.
  constructor(renderer) {
    this.renderer = renderer;
    this.camera = new THREE.Camera();
    this.areas = {};
    this.uniforms = { sunDir: { value: SUN }, camPos: { value: new THREE.Vector3() }, night: { value: 0.35 },
      time: { value: 0 } };
    this.facadeMat = new THREE.ShaderMaterial({ uniforms: this.uniforms, vertexShader: FACADE_VS,
      fragmentShader: FACADE_FS, side: THREE.FrontSide });
    this.streetMat = new THREE.ShaderMaterial({ uniforms: this.uniforms, vertexShader: STREET_VS,
      fragmentShader: STREET_FS, side: THREE.DoubleSide, polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -2 });
    this.groundMat = new THREE.ShaderMaterial({ uniforms: this.uniforms, vertexShader: STREET_VS, fragmentShader: STREET_FS });
    this.lastSim = 0;
  }

  async load(id) {
    if (this.areas[id]) return this.areas[id];
    const [osm, meta, grid] = await Promise.all([
      fetch(`/osm/${id}.json`).then(r => r.json()),
      fetch(`/data/flood/${id}.json`).then(r => r.json()),
      fetch(`/data/flood/${id}.bin`).then(r => r.arrayBuffer()),
    ]);
    const t0 = performance.now();
    const code = new Int16Array(grid);
    // Вся генерация геометрии — в воркере; сюда приходят готовые массивы вершин.
    const geo = await new Promise((resolve, reject) => {
      const w = new Worker("/static/city_worker.js", { type: "module" });
      w.onmessage = e => { resolve(e.data); w.terminate(); };
      w.onerror = e => { reject(new Error(e.message)); w.terminate(); };
      w.postMessage({ osm, meta, code: grid.slice(0) }, []);
    });
    const area = { id, osm, meta, code, scene: new THREE.Scene(), chunks: [], nBuildings: geo.nBuildings };
    const o = maplibregl.MercatorCoordinate.fromLngLat(osm.origin, 0);
    area.origin = o; area.meter = o.meterInMercatorCoordinateUnits();
    const toGeo = g => {
      const bg = new THREE.BufferGeometry();
      for (const [k, v] of Object.entries(g.attrs)) bg.setAttribute(k, new THREE.BufferAttribute(v, v.length / (g.attrs.position.length / 3)));
      bg.setIndex(new THREE.BufferAttribute(g.index, 1));
      bg.computeBoundingSphere();
      return bg;
    };
    let verts = 0;
    for (const c of geo.chunks) {
      const ch = { center: new THREE.Vector2(...c.center), b: null, s: null };
      if (c.b) { ch.b = new THREE.Mesh(toGeo(c.b), this.facadeMat); verts += c.b.attrs.position.length / 3; }
      if (c.s) { ch.s = new THREE.Mesh(toGeo(c.s), this.streetMat); verts += c.s.attrs.position.length / 3; }
      for (const m of [ch.b, ch.s]) if (m) { m.frustumCulled = true; area.scene.add(m); }
      area.chunks.push(ch);
    }
    area.groundMesh = new THREE.Mesh(toGeo(geo.ground), this.groundMat);
    area.groundMesh.geometry.computeVertexNormals();
    area.scene.add(area.groundMesh);
    this.buildFlood(area, geo);
    area.nVerts = verts;
    area.buildMs = performance.now() - t0;
    this.areas[id] = area;
    return area;
  }

  // GPU-затопление: текстура рельефа, пинг-понг глубины и расходов, сетка воды из воркера.
  buildFlood(area, geo) {
    const m = area.meta, n = m.nx * m.ny;
    const T = new Float32Array(n * 4), S = new Float32Array(n * 4);
    for (let j = 0; j < m.ny; j++) for (let i = 0; i < m.nx; i++) {
      const k = j * m.nx + i, c = area.code[k];
      const wall = c === m.building, bed = wall ? 1000 : c / 100;
      const water = !wall && c < 0;
      const edge = water && (i === 0 || j === 0 || i === m.nx - 1 || j === m.ny - 1);
      // R — отметка дна (здание — 1000, стена), G — земля (под зданием — вокруг него).
      T.set([bed, geo.under[k], edge ? 1 : 0, water ? 1 : 0], 4 * k);
      S[4 * k] = water ? -bed : 0;                       // уровень покоя 0 м
    }
    const tex = data => { const t = new THREE.DataTexture(data, m.nx, m.ny, THREE.RGBAFormat, THREE.FloatType);
      t.minFilter = t.magFilter = THREE.NearestFilter; t.needsUpdate = true; return t; };
    const rt = () => new THREE.WebGLRenderTarget(m.nx, m.ny, { type: THREE.FloatType, format: THREE.RGBAFormat,
      minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter, depthBuffer: false });
    const terrain = tex(T);
    const sim = { terrain, init: tex(S), state: [rt(), rt()], flux: [rt(), rt()], cur: 0, ready: false };
    const texel = new THREE.Vector2(1 / m.nx, 1 / m.ny);
    // Шаг по времени из условия устойчивости: dt < res / √(g·d), d до ~10 м.
    const dt = Math.min(0.5, 0.25 * m.res / Math.sqrt(9.81 * 10));
    const mk = fs => new THREE.ShaderMaterial({ vertexShader: SIM_VS, fragmentShader: fs, uniforms: {
      state: { value: null }, flux: { value: null }, terrain: { value: terrain }, texel: { value: texel },
      dt: { value: dt }, cell: { value: m.res }, sea: { value: 0 } } });
    sim.fluxMat = mk(FLUX_FS); sim.depthMat = mk(DEPTH_FS);
    sim.copyMat = new THREE.ShaderMaterial({ vertexShader: SIM_VS, uniforms: { src: { value: null } },
      fragmentShader: "uniform sampler2D src; varying vec2 vUv; void main() { gl_FragColor = texture2D(src, vUv); }" });
    sim.quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), sim.copyMat);
    sim.scene = new THREE.Scene(); sim.scene.add(sim.quad);
    sim.cam = new THREE.Camera();
    sim.dt = dt;
    area.sim = sim;
    const wg = new THREE.BufferGeometry();
    wg.setAttribute("position", new THREE.BufferAttribute(geo.water.attrs.position, 3));
    wg.setAttribute("cuv", new THREE.BufferAttribute(geo.water.attrs.cuv, 2));
    wg.setIndex(new THREE.BufferAttribute(geo.water.index, 1));
    sim.waterMat = new THREE.ShaderMaterial({ transparent: true, depthWrite: false,
      uniforms: { ...this.uniforms, state: { value: null }, terrain: { value: terrain }, texel: { value: texel },
        gridSize: { value: new THREE.Vector2(m.nx, m.ny) } },
      vertexShader: FLOODW_VS, fragmentShader: FLOODW_FS });
    const mesh = new THREE.Mesh(wg, sim.waterMat);
    mesh.frustumCulled = false;
    mesh.renderOrder = 3;
    area.scene.add(mesh);
    area.waterVerts = geo.water.attrs.position.length / 3;
  }

  pass(sim, mat, target) {
    sim.quad.material = mat;
    this.renderer.setRenderTarget(target);
    this.renderer.render(sim.scene, sim.cam);
  }

  // Расчёт затопления: не чаще SIM_HZ раз в секунду по SIM_SUBSTEPS шагов.
  simulate(area, seaLevel, running) {
    const sim = area.sim, r = this.renderer, now = performance.now();
    if (sim.ready && (!running || now - this.lastSim < 1000 / SIM_HZ)) return;
    this.lastSim = now;
    const prevTarget = r.getRenderTarget();
    if (!sim.ready) {
      sim.copyMat.uniforms.src.value = sim.init;
      this.pass(sim, sim.copyMat, sim.state[0]);
      sim.ready = true;
    }
    for (let s = 0; s < (running ? SIM_SUBSTEPS : 0); s++) {
      const a = sim.cur, b = 1 - a;
      sim.fluxMat.uniforms.state.value = sim.state[a].texture;
      sim.fluxMat.uniforms.flux.value = sim.flux[a].texture;
      this.pass(sim, sim.fluxMat, sim.flux[b]);
      sim.depthMat.uniforms.state.value = sim.state[a].texture;
      sim.depthMat.uniforms.flux.value = sim.flux[b].texture;
      sim.depthMat.uniforms.sea.value = seaLevel;
      this.pass(sim, sim.depthMat, sim.state[b]);
      sim.cur = b;
    }
    r.setRenderTarget(prevTarget);
    sim.waterMat.uniforms.state.value = sim.state[sim.cur].texture;
  }

  reset(area) { area.sim.ready = false; }

  // Отрисовка зоны: matrix — матрица MapLibre для единиц Меркатора.
  render(area, matrix) {
    const M = new THREE.Matrix4().fromArray(matrix).multiply(
      new THREE.Matrix4().makeTranslation(area.origin.x, area.origin.y, area.origin.z)
        .scale(new THREE.Vector3(area.meter, -area.meter, area.meter)));
    this.camera.projectionMatrix = M;
    this.camera.projectionMatrixInverse.copy(M).invert();
    const c = new THREE.Vector4(0, 0, 1, 0).applyMatrix4(this.camera.projectionMatrixInverse);
    const cam = this.uniforms.camPos.value.set(c.x / c.w, c.y / c.w, c.z / c.w);
    this.uniforms.time.value = performance.now() / 1000;
    // Уровни детализации по расстоянию от камеры до центра квартала (с учётом высоты).
    for (const ch of area.chunks) {
      const d = Math.hypot(ch.center.x - cam.x, ch.center.y - cam.y, cam.z);
      if (ch.s) ch.s.visible = d < LOD_STREETS + 360;           // 360 м — полуразмер квартала
      if (ch.b) ch.b.visible = d < LOD_BUILDINGS;
    }
    this.renderer.render(area.scene, this.camera);
  }
}
