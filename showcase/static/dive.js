// Свободная камера (6DOF) и подводный режим: дно EMODnet с микрорельефом и каустиками,
// вода на кольцевой сетке вокруг камеры с волнами Герстнера, окно Снеллиуса под водой.
// Рисуется в том же контексте WebGL, что и карта: город и затопление переиспользуются.
import * as THREE from "three";

const G = 9.81;
const SUN = new THREE.Vector3(0.8, 0.12, 0.26).normalize();
const WATER_RINGS = 220, WATER_SEGS = 512;     // 112 тыс. вершин: шаг ~5 см у камеры, км у горизонта
const BED_RINGS = 150, BED_SEGS = 256;
const N_WAVES = 8;

function radialGrid(rMin, rMax, rings, segs) {
  const pos = new Float32Array(rings * segs * 3);
  for (let i = 0; i < rings; i++) {
    const r = i === 0 ? 0 : rMin * Math.pow(rMax / rMin, (i - 1) / (rings - 2));
    for (let j = 0; j < segs; j++) {
      const a = (j / segs) * Math.PI * 2;
      pos.set([r * Math.cos(a), r * Math.sin(a), 0], 3 * (i * segs + j));
    }
  }
  const idx = [];
  for (let i = 0; i < rings - 1; i++) for (let j = 0; j < segs; j++) {
    const a = i * segs + j, b = i * segs + (j + 1) % segs, c = a + segs, d = b + segs;
    idx.push(a, c, b, b, c, d);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  g.setIndex(idx);
  return g;
}

// Общие функции: выборка рельефа (детально у Невской губы, иначе весь залив), шум, небо.
const TERRAIN_GLSL = `
  uniform sampler2D gulfTex; uniform vec4 gulfBox;     // отметка, м; рамка в локальных метрах (x0, y0, x1, y1)
  uniform sampler2D nevaTex; uniform vec4 nevaBox;     // R — отметка, м; G — до берега, м (> 0 — вода)
  uniform float time; uniform vec3 camPos; uniform vec3 sunDir; uniform float level; uniform float under;
  vec2 terrain(vec2 p) {                               // (отметка, расстояние до берега)
    vec2 un = (p - nevaBox.xy) / (nevaBox.zw - nevaBox.xy);
    if (all(greaterThan(un, vec2(0.002))) && all(lessThan(un, vec2(0.998)))) return texture2D(nevaTex, un).rg;
    vec2 ug = clamp((p - gulfBox.xy) / (gulfBox.zw - gulfBox.xy), 0.0, 1.0);
    float z = texture2D(gulfTex, ug).r;
    return vec2(z, -z * 40.0);
  }
  float hash12(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
  float vnoise(vec2 p) { vec2 i = floor(p), f = fract(p); vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash12(i), hash12(i + vec2(1, 0)), u.x), mix(hash12(i + vec2(0, 1)), hash12(i + vec2(1, 1)), u.x), u.y); }
  float fbm(vec2 p) { float s = 0.0, a = 0.5; for (int i = 0; i < 4; i++) { s += a * vnoise(p); p *= 2.03; a *= 0.5; } return s; }
  vec3 sky(vec3 r) {
    vec3 c = mix(vec3(0.80, 0.86, 0.92), vec3(0.22, 0.42, 0.74), pow(clamp(r.z, 0.0, 1.0), 0.55));
    c += vec3(1.0, 0.9, 0.7) * smoothstep(0.9993, 0.9997, dot(r, sunDir)) * 4.0 + vec3(1.0, 0.8, 0.5) * pow(max(dot(r, sunDir), 0.0), 12.0) * 0.2;
    return c;
  }
  // Каустики: два искажённых шумом узора, бегущих с волнами.
  float caustic(vec2 p) {
    vec2 q = p * 0.6 + vec2(fbm(p * 0.15 + time * 0.2), fbm(p * 0.15 - time * 0.17)) * 3.0;
    float a = abs(sin(q.x + time * 1.3) * sin(q.y - time * 0.9));
    float b = abs(sin(q.x * 1.7 - time * 0.7) * sin(q.y * 1.3 + time * 1.1));
    return pow(1.0 - min(a, b), 8.0);
  }
  const vec3 FOG_UNDER = vec3(0.05, 0.22, 0.24);
  const vec3 ABSORB = vec3(0.45, 0.10, 0.07);          // ослабление света водой, 1/м (Бугер — Ламберт)
`;

const BED_VS = TERRAIN_GLSL + `
  varying vec3 vP; varying vec2 vT;
  void main() {
    vec3 p = position + vec3(camPos.xy, 0.0);
    vec2 t = terrain(p.xy);
    // Микрорельеф: песчаная рябь и бугры на дне, только вблизи (дальше не видно).
    float near = smoothstep(400.0, 30.0, length(position.xy));
    float rip = 0.03 * sin(p.x * 1.9 + 2.5 * fbm(p.xy * 0.08)) + 0.15 * (fbm(p.xy * 0.05) - 0.5);
    p.z = t.x + (t.x < 0.0 ? rip * near : 0.0);
    vP = p; vT = t;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(p, 1.0);
  }`;
const BED_FS = TERRAIN_GLSL + `
  varying vec3 vP; varying vec2 vT;
  void main() {
    vec3 dx = dFdx(vP), dy = dFdy(vP);
    vec3 n = normalize(cross(dx, dy));
    if (n.z < 0.0) n = -n;
    float slope = 1.0 - n.z;
    float depth = level - vP.z;
    vec3 c;
    if (depth > 0.0) {
      // Дно: песок с рябью у берега, ил на глубине, камни на крутых склонах.
      float rip = 0.5 + 0.5 * sin(vP.x * 1.9 + 2.5 * fbm(vP.xy * 0.08));
      vec3 sand = mix(vec3(0.62, 0.55, 0.40), vec3(0.74, 0.67, 0.50), rip * 0.5 + 0.3 * fbm(vP.xy * 0.5));
      vec3 silt = mix(vec3(0.24, 0.24, 0.17), vec3(0.32, 0.30, 0.22), fbm(vP.xy * 0.2));
      float rocks = smoothstep(0.55, 0.7, fbm(vP.xy * 0.12)) + smoothstep(0.08, 0.25, slope);
      vec3 rock = vec3(0.36, 0.35, 0.33) * (0.7 + 0.5 * vnoise(vP.xy * 2.0));
      c = mix(sand, silt, smoothstep(3.0, 10.0, depth));
      c = mix(c, rock, clamp(rocks, 0.0, 1.0) * 0.8);
      float lit = 0.35 + 0.65 * max(dot(n, sunDir), 0.0);
      c *= lit;
      c += vec3(0.9, 0.95, 0.85) * caustic(vP.xy) * exp(-depth / 5.0) * 0.55;   // каустики
      c *= exp(-ABSORB * depth);                                                 // свет теряется в толще
    } else {
      // Суша (в подводном режиме нет снимка): пляж, трава, лес, камень.
      float e = vP.z - level;
      vec3 grass = mix(vec3(0.30, 0.40, 0.20), vec3(0.20, 0.30, 0.15), fbm(vP.xy * 0.02));
      c = mix(vec3(0.72, 0.66, 0.50), grass, smoothstep(0.6, 2.0, e));
      c = mix(c, vec3(0.45, 0.44, 0.42), smoothstep(0.15, 0.35, slope));
      c *= 0.45 + 0.6 * max(dot(n, sunDir), 0.0);
    }
    float dist = length(camPos - vP);
    if (under > 0.5) c = mix(c, FOG_UNDER, 1.0 - exp(-dist / 14.0));           // мутная балтийская вода
    else c = mix(c, vec3(0.80, 0.86, 0.92), (1.0 - exp(-dist / 25000.0)) * 0.85);
    gl_FragColor = vec4(c, 1.0);
  }`;

// Вода: 8 волн Герстнера. Частота каждой — по дисперсии ω² = g·k·tanh(k·h) для глубины у камеры;
// на мелководье амплитуда растёт (шоалинг) и ограничена обрушением H/h ≤ 0,78.
const WATER_COMMON = TERRAIN_GLSL + `
  uniform vec4 waves[${N_WAVES}];      // (dx, dy, k, амплитуда при глубокой воде)
  uniform float omega[${N_WAVES}];
  uniform vec4 cityBox;
  uniform sampler2D cityTerrain; uniform vec4 cityTexBox; uniform float hasCity;
  float shoal(float k, float h) {                      // Ks = √(cg0 / cg(h)), kh по Экарту
    float kh0 = k * h, kh = kh0 / sqrt(max(tanh(kh0), 1e-3));
    float n = 0.5 * (1.0 + 2.0 * kh / sinh(min(2.0 * kh, 40.0)));
    float cg = n * sqrt(tanh(kh) / max(kh, 1e-3)) * sqrt(G_ * h);
    float cg0 = 0.5 * sqrt(G_ / k);
    return clamp(sqrt(cg0 / max(cg, 1e-3)), 0.6, 2.2);
  }
`.replace(/G_/g, "9.81");

const WATER_VS = WATER_COMMON + `
  varying vec3 vP; varying vec3 vN; varying float vFoam; varying float vDepth; varying float vShore;
  void main() {
    vec3 base = position + vec3(camPos.xy, 0.0);
    vec2 t = terrain(base.xy);
    float h = max(level - t.x, 0.05);
    vec3 p = vec3(base.xy, level);
    vec3 tx = vec3(1.0, 0.0, 0.0), ty = vec3(0.0, 1.0, 0.0);
    float amp = 0.0, jac = 1.0;
    float fade = smoothstep(6000.0, 800.0, length(position.xy));         // вдали волны не видны
    for (int i = 0; i < ${N_WAVES}; i++) {
      vec2 d = waves[i].xy; float k = waves[i].z;
      float A = waves[i].w * shoal(k, h) * fade;
      float ph = k * dot(d, base.xy) - omega[i] * time;
      float Q = min(0.8 / (k * A * float(${N_WAVES}) + 1e-4), 1.0);   // крутизна без петель
      float s = sin(ph), c = cos(ph);
      p.xy += Q * A * d * c;
      p.z += A * s;
      amp += A;
      float wa = k * A;
      tx += vec3(-Q * d.x * d.x * wa * s, -Q * d.x * d.y * wa * s, d.x * wa * c);
      ty += vec3(-Q * d.x * d.y * wa * s, -Q * d.y * d.y * wa * s, d.y * wa * c);
      jac -= Q * wa * s;
    }
    // Обрушение: высота волны не больше 0,78 глубины — лишнее срезаем, там пена.
    float limit = 0.39 * h;
    float over = max(amp - limit, 0.0);
    p.z = level + (p.z - level) * min(1.0, limit / max(amp, 1e-4));
    vN = normalize(cross(tx, ty));
    vFoam = clamp((1.0 - jac) * 1.4 - 0.2, 0.0, 1.0) + clamp(over / max(limit, 0.05), 0.0, 1.0);
    vDepth = h; vShore = t.y;
    vP = p;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(p, 1.0);
  }`;
const WATER_FS = WATER_COMMON + `
  varying vec3 vP; varying vec3 vN; varying float vFoam; varying float vDepth; varying float vShore;
  void main() {
    if (vP.x > cityBox.x && vP.x < cityBox.z && vP.y > cityBox.y && vP.y < cityBox.w) discard;  // там затопление города
    if (vShore < -2.0) discard;                                                  // суша
    // Мелкая рябь в нормали (то, что сетка уже не берёт), видна только вблизи.
    float mpp = max(length(fwidth(vP.xy)), 0.002);
    vec2 g = vec2(0.0);
    for (int i = 0; i < 6; i++) {
      float fi = float(i), L = 1.2 * pow(0.6, fi), k = 6.2831 / L, w = sqrt(9.81 * k);
      vec2 d = normalize(vec2(cos(fi * 2.3 + 0.4), sin(fi * 2.3 + 0.4)) + vec2(1.2, 0.2));
      g += d * 0.10 * cos(k * dot(d, vP.xy) - w * time + fi * 1.7) * smoothstep(8.0, 20.0, L / mpp);   // без муара
    }
    vec3 n = normalize(normalize(vN) + vec3(-g, 0.0));
    vec3 V = normalize(camPos - vP);
    float dist = length(camPos - vP);
    // Пена: гребни Герстнера, обрушение на мелком, берег, стены набережных и домов.
    float wall = 0.0;
    if (hasCity > 0.5) {
      vec2 uc = (vP.xy - cityTexBox.xy) / (cityTexBox.zw - cityTexBox.xy);
      for (int k = 0; k < 4; k++) {
        vec2 o = vec2(k == 0 ? 6.0 : k == 1 ? -6.0 : 0.0, k == 2 ? 6.0 : k == 3 ? -6.0 : 0.0) / (cityTexBox.zw - cityTexBox.xy);
        wall += step(500.0, texture2D(cityTerrain, uc + o).r);
      }
    }
    float shoreFoam = smoothstep(18.0, 2.0, vShore) + wall * 0.3;
    float foamMask = clamp(vFoam + shoreFoam, 0.0, 1.0) * smoothstep(0.35, 0.75, fbm(vP.xy * 0.7 + time * 0.3));
    if (under < 0.5) {
      vec3 R = reflect(-V, n);
      float F = 0.02 + 0.98 * pow(1.0 - clamp(dot(n, V), 0.0, 1.0), 5.0);
      vec3 body = mix(vec3(0.03, 0.16, 0.20), vec3(0.08, 0.34, 0.33), exp(-vDepth / 6.0));
      body += vec3(0.05, 0.22, 0.18) * clamp((vP.z - level) * 1.5, 0.0, 1.0);    // гребень просвечивает
      vec3 c = mix(body, sky(R), F);
      c = mix(c, vec3(0.95, 0.97, 0.98), foamMask * 0.9);
      c = mix(c, vec3(0.80, 0.86, 0.92), (1.0 - exp(-dist / 25000.0)) * 0.85);
      // Прозрачность: на мелком видно дно, на глубине — нет.
      float alpha = clamp(1.0 - exp(-vDepth * 0.45), 0.35, 0.96) + F * 0.3 + foamMask;
      gl_FragColor = vec4(c, clamp(alpha, 0.0, 1.0));
    } else {
      // Снизу: в конусе 48,6° (окно Снеллиуса) видно небо, за ним — полное отражение.
      vec3 up = -V;                                       // луч от камеры к поверхности
      float cosI = clamp(dot(up, n), 0.0, 1.0);
      float window = smoothstep(0.62, 0.68, cosI);        // cos 48,6° ≈ 0,661
      vec3 c = mix(FOG_UNDER * 0.6, sky(refract(up, -n, 1.333)) * 0.9 + 0.1, window);
      c = mix(c, vec3(0.9), foamMask * 0.6);
      c = mix(c, FOG_UNDER, 1.0 - exp(-dist / 14.0));
      gl_FragColor = vec4(c, 1.0);
    }
  }`;

const SKY_VS = `varying vec2 vUv; void main() { vUv = uv; gl_Position = vec4(position.xy, 0.9999, 1.0); }`;
const SKY_FS = `
  uniform mat4 invViewProj; uniform vec3 sunDir; uniform float under; varying vec2 vUv;
  void main() {
    vec4 a = invViewProj * vec4(vUv * 2.0 - 1.0, -1.0, 1.0), b = invViewProj * vec4(vUv * 2.0 - 1.0, 1.0, 1.0);
    vec3 r = normalize(b.xyz / b.w - a.xyz / a.w);
    vec3 c = mix(vec3(0.80, 0.86, 0.92), vec3(0.22, 0.42, 0.74), pow(clamp(r.z, 0.0, 1.0), 0.55));
    c += vec3(1.0, 0.9, 0.7) * smoothstep(0.9993, 0.9997, dot(r, sunDir)) * 4.0 + vec3(1.0, 0.8, 0.5) * pow(max(dot(r, sunDir), 0.0), 12.0) * 0.2;
    if (r.z < 0.0) c = mix(vec3(0.80, 0.86, 0.92), vec3(0.2, 0.3, 0.33), clamp(-r.z * 4.0, 0.0, 1.0));
    if (under > 0.5) c = mix(vec3(0.05, 0.22, 0.24), vec3(0.02, 0.08, 0.10), clamp(-r.z, 0.0, 1.0));
    gl_FragColor = vec4(c, 1.0);
  }`;

export class Dive {
  constructor(renderer) {
    this.renderer = renderer;
    this.active = false;
    this.camera = new THREE.PerspectiveCamera(65, 1, 0.05, 80000);
    this.camera.up.set(0, 0, 1);
    this.scene = new THREE.Scene();
    this.speed = 25;
    this.loaded = false;
  }

  async load() {
    if (this.loaded) return;
    const [gj, gb, nj, nb, ns] = await Promise.all([
      fetch("/data/seabed/gulf.json").then(r => r.json()), fetch("/data/seabed/gulf.bin").then(r => r.arrayBuffer()),
      fetch("/data/seabed/neva.json").then(r => r.json()), fetch("/data/seabed/neva.bin").then(r => r.arrayBuffer()),
      fetch("/data/seabed/neva_sdf.bin").then(r => r.arrayBuffer()),
    ]);
    const g16 = new Int16Array(gb), gf = new Float32Array(g16.length);
    for (let i = 0; i < g16.length; i++) gf[i] = g16[i] / 10;
    const n16 = new Int16Array(nb), s16 = new Int16Array(ns), nf = new Float32Array(n16.length * 2);
    for (let i = 0; i < n16.length; i++) { nf[2 * i] = n16[i] / 100; nf[2 * i + 1] = s16[i] / 10; }
    const mk = (data, w, h, fmt) => { const t = new THREE.DataTexture(data, w, h, fmt, THREE.FloatType);
      t.minFilter = t.magFilter = THREE.LinearFilter; t.needsUpdate = true; return t; };
    this.gulf = { meta: gj, tex: mk(gf, gj.nx, gj.ny, THREE.RedFormat), data: gf };
    this.neva = { meta: nj, tex: mk(nf, nj.nx, nj.ny, THREE.RGFormat), data: nf };
    this.uniforms = {
      gulfTex: { value: this.gulf.tex }, gulfBox: { value: new THREE.Vector4() },
      nevaTex: { value: this.neva.tex }, nevaBox: { value: new THREE.Vector4() },
      time: { value: 0 }, camPos: { value: new THREE.Vector3() }, sunDir: { value: SUN },
      level: { value: 0 }, under: { value: 0 },
      waves: { value: Array.from({ length: N_WAVES }, () => new THREE.Vector4()) },
      omega: { value: new Array(N_WAVES).fill(0) },
      cityBox: { value: new THREE.Vector4(1e9, 1e9, -1e9, -1e9) },
      cityTerrain: { value: null }, cityTexBox: { value: new THREE.Vector4() }, hasCity: { value: 0 },
    };
    const bed = new THREE.Mesh(radialGrid(1, 60000, BED_RINGS, BED_SEGS),
      new THREE.ShaderMaterial({ uniforms: this.uniforms, vertexShader: BED_VS, fragmentShader: BED_FS }));
    const water = new THREE.Mesh(radialGrid(0.25, 60000, WATER_RINGS, WATER_SEGS),
      new THREE.ShaderMaterial({ uniforms: this.uniforms, vertexShader: WATER_VS, fragmentShader: WATER_FS,
        transparent: true, side: THREE.DoubleSide, depthWrite: false }));
    for (const m of [bed, water]) m.frustumCulled = false;
    water.renderOrder = 2;
    this.skyMat = new THREE.ShaderMaterial({ vertexShader: SKY_VS, fragmentShader: SKY_FS, depthWrite: false, depthTest: false,
      uniforms: { invViewProj: { value: new THREE.Matrix4() }, sunDir: { value: SUN }, under: this.uniforms.under } });
    this.sky = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), this.skyMat);
    this.sky.frustumCulled = false;
    this.skyScene = new THREE.Scene(); this.skyScene.add(this.sky);
    this.scene.add(bed, water);
    this.bed = bed; this.water = water;
    this.loaded = true;
  }

  // Локальные метры относительно точки входа (свой центр — без дрожания float32).
  setOrigin(lon, lat) {
    this.origin = maplibregl.MercatorCoordinate.fromLngLat([lon, lat], 0);
    this.meter = this.origin.meterInMercatorCoordinateUnits();
    this.lon0 = lon; this.lat0 = lat;
    const box = (b) => {
      const [x0, y0] = this.local(b[0], b[1]), [x1, y1] = this.local(b[2], b[3]);
      return new THREE.Vector4(x0, y0, x1, y1);
    };
    const gb = this.gulf.meta.bbox, nb = this.neva.meta.bbox;
    this.uniforms.gulfBox.value.copy(box([gb[0], gb[1], gb[2], gb[3]]));
    this.uniforms.nevaBox.value.copy(box([nb[0], nb[1], nb[2], nb[3]]));
  }
  local(lon, lat) {
    const m = maplibregl.MercatorCoordinate.fromLngLat([lon, lat], 0);
    return [(m.x - this.origin.x) / this.meter, -(m.y - this.origin.y) / this.meter];
  }
  lonLat(x, y) {
    return new maplibregl.MercatorCoordinate(this.origin.x + x * this.meter, this.origin.y - y * this.meter, 0).toLngLat();
  }

  // Спектр волн по ветру: пик по Пирсону — Московицу, Hs ≈ 0,012·U² (ограничено разгоном).
  setWind(U, dirDeg) {
    U = Math.max(U, 5);
    const Lp = Math.min(2 * Math.PI * U * U / (0.769 * G), 70);
    const Hs = Math.min(0.012 * U * U, 3);
    const to = (dirDeg + 180) * Math.PI / 180;                         // ветер «откуда» → волны «куда»
    const base = new THREE.Vector2(Math.sin(to), Math.cos(to));
    const rel = [1.0, 0.78, 0.6, 0.45, 0.32, 0.22, 0.14, 0.08];
    const spread = [0, 0.35, -0.4, 0.7, -0.75, 1.1, -1.2, 0.5];
    const sum = rel.reduce((a, r) => a + r * r, 0);
    this.waveSet = rel.map((r, i) => {
      const L = Lp * r, k = 2 * Math.PI / L, a = spread[i];
      const d = new THREE.Vector2(base.x * Math.cos(a) - base.y * Math.sin(a), base.x * Math.sin(a) + base.y * Math.cos(a));
      return { d, k, A: 0.5 * Hs * r / Math.sqrt(sum) };
    });
    this.waveSet.forEach((w, i) => this.uniforms.waves.value[i].set(w.d.x, w.d.y, w.k, w.A));
    this.Hs = Hs;
  }
  // Частоты по дисперсии с конечной глубиной у камеры: ω² = g·k·tanh(k·h).
  setDepth(h) {
    h = Math.max(h, 0.3);
    this.waveSet.forEach((w, i) => { this.uniforms.omega.value[i] = Math.sqrt(G * w.k * Math.tanh(w.k * h)); });
  }
  // Высота воды под камерой — по тем же формулам, что в шейдере (для «под водой»).
  surfaceAt(x, y, t, level) {
    let z = level;
    this.waveSet.forEach((w, i) => { z += w.A * Math.sin(w.k * (w.d.x * x + w.d.y * y) - this.uniforms.omega.value[i] * t); });
    return z;
  }

  // Отметка дна/суши (м) под точкой: детально у Невской губы, иначе весь залив.
  elevationAt(lon, lat) {
    for (const [src, stride] of [[this.neva, 2], [this.gulf, 1]]) {
      const [w, s, e, n] = src.meta.bbox, nx = src.meta.nx, ny = src.meta.ny;
      if (lon <= w || lon >= e || lat <= s || lat >= n) continue;
      const i = Math.floor((lon - w) / (e - w) * nx), j = Math.floor((lat - s) / (n - s) * ny);
      return src.data[(j * nx + i) * stride];
    }
    return -20;
  }

  enter(lon, lat, bearing, altitude) {
    this.setOrigin(lon, lat);
    this.camera.position.set(0, 0, altitude);
    const yaw = (90 - bearing) * Math.PI / 180;                       // азимут → угол от востока
    const target = new THREE.Vector3(Math.cos(yaw), Math.sin(yaw), -0.25).add(this.camera.position);
    this.camera.lookAt(target);
    this.active = true;
  }

  // Полёт 6DOF: мышь — рысканье и тангаж в осях камеры, Q/E — крен, WASD/R/F — сдвиг.
  look(dx, dy) {
    const q = new THREE.Quaternion();
    q.setFromAxisAngle(new THREE.Vector3(0, 1, 0), -dx * 0.0022); this.camera.quaternion.multiply(q);
    q.setFromAxisAngle(new THREE.Vector3(1, 0, 0), -dy * 0.0022); this.camera.quaternion.multiply(q);
  }
  move(keys, dt) {
    const v = this.speed * (keys.has("shift") ? 8 : 1) * dt;
    const m = new THREE.Vector3(
      (keys.has("d") ? 1 : 0) - (keys.has("a") ? 1 : 0),
      (keys.has("r") ? 1 : 0) - (keys.has("f") ? 1 : 0),
      (keys.has("s") ? 1 : 0) - (keys.has("w") ? 1 : 0)).multiplyScalar(v);
    this.camera.position.add(m.applyQuaternion(this.camera.quaternion));
    const roll = ((keys.has("e") ? 1 : 0) - (keys.has("q") ? 1 : 0)) * 1.2 * dt;
    if (roll) this.camera.quaternion.multiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), -roll));
  }

  // Матрица «камера × Меркатор» для слоёв, которые живут в единицах Меркатора (город).
  mercatorMatrix() {
    const toLocal = new THREE.Matrix4().makeScale(1 / this.meter, -1 / this.meter, 1 / this.meter)
      .multiply(new THREE.Matrix4().makeTranslation(-this.origin.x, -this.origin.y, 0));
    return new THREE.Matrix4().multiplyMatrices(this.camera.projectionMatrix, this.camera.matrixWorldInverse).multiply(toLocal);
  }

  render(width, height, t, level, city) {
    const cam = this.camera;
    cam.aspect = width / height; cam.updateProjectionMatrix(); cam.updateMatrixWorld();
    const u = this.uniforms;
    u.time.value = t; u.camPos.value.copy(cam.position); u.level.value = level;
    const surf = this.surfaceAt(cam.position.x, cam.position.y, t, level);
    this.underwater = cam.position.z < surf - 0.02;
    u.under.value = this.underwater ? 1 : 0;
    if (city) {
      const [x0, y0] = this.local(city.bbox[1], city.bbox[0]), [x1, y1] = this.local(city.bbox[3], city.bbox[2]);
      u.cityBox.value.set(x0, y0, x1, y1);
      u.cityTerrain.value = city.terrain; u.cityTexBox.value.set(x0, y0, x1, y1); u.hasCity.value = 1;
    } else { u.cityBox.value.set(1e9, 1e9, -1e9, -1e9); u.hasCity.value = 0; }
    this.skyMat.uniforms.invViewProj.value.multiplyMatrices(cam.projectionMatrix, cam.matrixWorldInverse).invert();
    const r = this.renderer;
    r.render(this.skyScene, cam);
    r.clearDepth();
    r.render(this.scene, cam);
    return surf;
  }
}
