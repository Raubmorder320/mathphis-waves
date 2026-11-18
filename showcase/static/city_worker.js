// Сборка геометрии детального города в отдельном потоке: главный поток только
// создаёт BufferGeometry из готовых массивов (без генерации при движении камеры).
// Кварталы 500 × 500 м: у каждого свой буфер зданий и улиц — работает отсечение вне экрана.
import { ShapeUtils, Vector2, Color } from "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js";

const CHUNK = 500;           // м
const CURB = 0.18;           // м — бордюр
const PALETTE = [
  [0.86, 0.70, 0.42], [0.90, 0.82, 0.64], [0.72, 0.80, 0.62], [0.86, 0.64, 0.58],
  [0.62, 0.74, 0.82], [0.78, 0.76, 0.72], [0.93, 0.88, 0.78], [0.80, 0.56, 0.36],
];
const ROOFS = [[0.33, 0.42, 0.36], [0.42, 0.42, 0.44], [0.50, 0.30, 0.22], [0.30, 0.32, 0.34]];
const CSS = { white: "#efefea", yellow: "#d9b45c", beige: "#d8c4a0", grey: "#9a9a98", gray: "#9a9a98",
  brown: "#8a5a3c", red: "#a8483a", pink: "#d49a90", green: "#8fa880", blue: "#8aa2bf", orange: "#d08a4a" };
const hash = n => { const x = Math.sin(n * 127.1) * 43758.5453; return x - Math.floor(x); };
function colour(c) {
  if (!c) return null;
  try { const k = String(c).toLowerCase(); const col = new Color(CSS[k] || k); return [col.r, col.g, col.b]; } catch { return null; }
}

// Меркатор как в MapLibre: метры зоны → локальные метры относительно её угла.
function mercator(lon, lat) {
  return [(lon + 180) / 360, (1 - Math.log(Math.tan(Math.PI / 4 + lat * Math.PI / 360)) / Math.PI) / 2];
}

class Buf {
  constructor() { this.a = {}; this.idx = []; this.n = 0; }
  push(name, ...v) { (this.a[name] ||= []).push(...v); }
  pack() {
    const out = { index: new Uint32Array(this.idx), attrs: {} };
    for (const [k, v] of Object.entries(this.a)) out.attrs[k] = new Float32Array(v);
    return out;
  }
}

onmessage = async ({ data }) => {
  // Загрузка и разбор данных зоны — тоже здесь, чтобы главный поток не стоял на JSON.
  const [osm, meta, grid] = await Promise.all([
    fetch(`/osm/${data.id}.json`).then(r => r.json()),
    fetch(`/data/flood/${data.id}.json`).then(r => r.json()),
    fetch(`/data/flood/${data.id}.bin`).then(r => r.arrayBuffer()),
  ]);
  const code = new Int16Array(grid);
  const [ox, oy] = mercator(osm.origin[0], osm.origin[1]);
  const meter = 1 / (2 * Math.PI * 6378137 * Math.cos(osm.origin[1] * Math.PI / 180));
  const toLocal = (x, y) => {
    const [mx, my] = mercator(osm.origin[0] + x / osm.kx, osm.origin[1] + y / osm.ky);
    return [(mx - ox) / meter, -(my - oy) / meter];
  };
  const ground = (x, y) => {
    const i = Math.min(Math.max(Math.round(x / meta.res), 0), meta.nx - 1);
    const j = Math.min(Math.max(Math.round(y / meta.res), 0), meta.ny - 1);
    const v = code[j * meta.nx + i];
    if (v !== meta.building) return v / 100;
    for (let r = 1; r < 8; r++) for (const [di, dj] of [[r, 0], [-r, 0], [0, r], [0, -r]]) {
      const ii = i + di, jj = j + dj;
      if (ii < 0 || jj < 0 || ii >= meta.nx || jj >= meta.ny) continue;
      const w = code[jj * meta.nx + ii];
      if (w !== meta.building) return Math.max(w / 100, 0.6);
    }
    return 2;
  };
  const [SX, SY] = osm.size;                 // размер зоны, м: координаты текстур расчёта и снимка
  const chunks = new Map();
  const chunk = (x, y) => {
    const key = `${Math.floor(x / CHUNK)}_${Math.floor(y / CHUNK)}`;
    if (!chunks.has(key)) chunks.set(key, { b: new Buf(), s: new Buf() });
    return chunks.get(key);
  };

  // Здания: стены (без пола) и крыша.
  let nb = 0;
  for (const b of osm.buildings) {
    const ring = b.p;
    if (ring.length < 3) continue;
    let s2 = 0, cx = 0, cy = 0;
    for (let i = 0; i < ring.length; i++) {
      const [x1, y1] = ring[i], [x2, y2] = ring[(i + 1) % ring.length];
      s2 += x1 * y2 - x2 * y1; cx += x1; cy += y1;
    }
    const pts = s2 > 0 ? ring : ring.slice().reverse();
    let gmin = Infinity;
    for (const [x, y] of pts) gmin = Math.min(gmin, ground(x, y));
    const base = gmin + b.b, top = gmin + b.h, H = b.h - b.b;
    const seed = hash(nb + 1);
    const industrial = /industrial|warehouse|garage|shed|hangar/.test(b.k);
    const col = colour(b.c) || (industrial ? [0.62, 0.36, 0.28] : PALETTE[Math.floor(seed * PALETTE.length)]);
    const levels = Math.max(1, Math.round(H / 3.3));
    const floorH = Math.min(Math.max(H / levels, 2.8), 4.6);
    const spacing = industrial ? 4.2 : b.hist ? 3.4 : 3.0;
    const kind = industrial ? 1 : 0;
    const B = chunk(cx / ring.length, cy / ring.length).b;
    const P = pts.map(([x, y]) => toLocal(x, y));
    let u = 0;
    for (let i = 0; i < P.length; i++) {
      const [ax, ay] = P[i], [bx, by] = P[(i + 1) % P.length];
      const len = Math.hypot(bx - ax, by - ay);
      if (len < 0.05) continue;
      const tx = (bx - ax) / len, ty = (by - ay) / len;
      const v0 = B.n;
      B.push("position", ax, ay, base, bx, by, base, bx, by, top, ax, ay, top);
      const [ua, va] = pts[i], [ub, vb] = pts[(i + 1) % pts.length];     // метры зоны → текстура расчёта
      B.push("cuv", ua / SX, va / SY, ub / SX, vb / SY, ub / SX, vb / SY, ua / SX, va / SY);
      for (let k = 0; k < 4; k++) {
        B.push("normal", ty, -tx, 0); B.push("wtan", tx, ty, 0); B.push("fcolor", ...col);
        B.push("fparam", floorH, spacing, H, kind);
      }
      B.push("wuv", u, b.b, u + len, b.b, u + len, b.h, u, b.h);
      B.idx.push(v0, v0 + 1, v0 + 2, v0, v0 + 2, v0 + 3);
      B.n += 4;
      u += len;
    }
    let tri = [];
    try { tri = ShapeUtils.triangulateShape(P.map(([x, y]) => new Vector2(x, y)), []); } catch { tri = []; }
    const r0 = B.n;
    const rc = colour(b.rc) || ROOFS[Math.floor(hash(nb + 7) * ROOFS.length)];
    for (let q = 0; q < P.length; q++) {
      const [x, y] = P[q];
      B.push("position", x, y, top); B.push("normal", 0, 0, 1); B.push("wtan", 1, 0, 0); B.push("wuv", 0, 0);
      B.push("cuv", pts[q][0] / SX, pts[q][1] / SY);
      B.push("fcolor", ...rc); B.push("fparam", floorH, spacing, H, 2);
    }
    B.n += P.length;
    for (const [a, bb, c] of tri) B.idx.push(r0 + a, r0 + bb, r0 + c);
    nb++;
  }

  // Улицы: проезжая часть, тротуары на CURB выше, грань бордюра; набережные — гранит.
  const quad = (S, a, b, c, d, n, k, uv) => {
    const v0 = S.n;
    S.push("position", ...a, ...b, ...c, ...d);
    for (let i = 0; i < 4; i++) { S.push("normal", ...n); S.push("kind", k); }
    S.push("suv", ...uv);
    S.idx.push(v0, v0 + 1, v0 + 2, v0, v0 + 2, v0 + 3);
    S.n += 4;
  };
  for (const r of osm.roads) {
    if (r.tu) continue;
    const half = r.w / 2, sw = r.sw;
    let u = 0;
    for (let i = 0; i < r.p.length - 1; i++) {
      const [axz, ayz] = r.p[i], [bxz, byz] = r.p[i + 1];
      const [ax, ay] = toLocal(axz, ayz), [bx, by] = toLocal(bxz, byz);
      const len = Math.hypot(bx - ax, by - ay);
      if (len < 0.1) continue;
      const S = chunk(0.5 * (axz + bxz), 0.5 * (ayz + byz)).s;
      const nx = -(by - ay) / len, ny = (bx - ax) / len;
      const za = r.br ? 6 : ground(axz, ayz) + 0.06, zb = r.br ? 6 : ground(bxz, byz) + 0.06;
      const off = (x, y, s, z) => [x + nx * s, y + ny * s, z];
      quad(S, off(ax, ay, -half, za), off(bx, by, -half, zb), off(bx, by, half, zb), off(ax, ay, half, za), [0, 0, 1], 0,
        [u, -half, u + len, -half, u + len, half, u, half]);
      if (sw > 0 && !r.br) for (const side of [-1, 1]) {
        const s0 = side * half, s1 = side * (half + sw);
        quad(S, off(ax, ay, s0, za + CURB), off(bx, by, s0, zb + CURB), off(bx, by, s1, zb + CURB), off(ax, ay, s1, za + CURB),
          [0, 0, 1], 1, [u, 0, u + len, 0, u + len, 1, u, 1]);
        quad(S, off(ax, ay, s0, za), off(bx, by, s0, zb), off(bx, by, s0, zb + CURB), off(ax, ay, s0, za + CURB),
          [-nx * side, -ny * side, 0], 2, [u, 0, u + len, 0, u + len, 1, u, 1]);
      }
      u += len;
    }
  }
  for (const q of osm.quays) {
    let u = 0;
    for (let i = 0; i < q.p.length - 1; i++) {
      const [axz, ayz] = q.p[i], [bxz, byz] = q.p[i + 1];
      const [ax, ay] = toLocal(axz, ayz), [bx, by] = toLocal(bxz, byz);
      const len = Math.hypot(bx - ax, by - ay);
      if (len < 0.1) continue;
      const S = chunk(0.5 * (axz + bxz), 0.5 * (ayz + byz)).s;
      const nx = -(by - ay) / len, ny = (bx - ax) / len;
      const za = Math.max(ground(axz, ayz), 0.6) + meta.parapet, zb = Math.max(ground(bxz, byz), 0.6) + meta.parapet;
      for (const s of [0.35, -0.35]) quad(S, [ax + nx * s, ay + ny * s, -1.5], [bx + nx * s, by + ny * s, -1.5],
        [bx + nx * s, by + ny * s, zb], [ax + nx * s, ay + ny * s, za], [nx * Math.sign(s), ny * Math.sign(s), 0], 2,
        [u, 0, u + len, 0, u + len, 1, u, 1]);
      quad(S, [ax - nx * 0.35, ay - ny * 0.35, za], [bx - nx * 0.35, by - ny * 0.35, zb],
        [bx + nx * 0.35, by + ny * 0.35, zb], [ax + nx * 0.35, ay + ny * 0.35, za], [0, 0, 1], 2,
        [u, 0, u + len, 0, u + len, 1, u, 1]);
      u += len;
    }
  }

  // Земля зоны — грубая сетка (≤ 60 тыс. вершин): рельеф сглаженный, детали дают улицы.
  const gstep = Math.max(1, Math.ceil(Math.sqrt(meta.nx * meta.ny / 60000)));
  const G = new Buf();
  const gx = Math.floor(meta.nx / gstep), gy = Math.floor(meta.ny / gstep);
  for (let j = 0; j < gy; j++) for (let i = 0; i < gx; i++) {
    const x = (i * gstep + 0.5) * meta.res, y = (j * gstep + 0.5) * meta.res;
    const c = code[j * gstep * meta.nx + i * gstep];
    const [lx, ly] = toLocal(x, y);
    G.push("position", lx, ly, c === meta.building ? ground(x, y) : c / 100);
    G.push("normal", 0, 0, 1); G.push("suv", x / SX, y / SY); G.push("kind", c < 0 ? 4 : 3);
  }
  for (let j = 0; j < gy - 1; j++) for (let i = 0; i < gx - 1; i++) {
    const a = j * gx + i; G.idx.push(a, a + 1, a + gx, a + 1, a + gx + 1, a + gx);
  }

  // Сетка воды затопления (≤ 100 тыс. вершин) с координатами в текстуре расчёта.
  const wstep = Math.max(1, Math.ceil(Math.sqrt(meta.nx * meta.ny / 100000)));
  const W = new Buf();
  const wx = Math.floor(meta.nx / wstep), wy = Math.floor(meta.ny / wstep);
  for (let j = 0; j < wy; j++) for (let i = 0; i < wx; i++) {
    const [lx, ly] = toLocal((i * wstep + 0.5) * meta.res, (j * wstep + 0.5) * meta.res);
    W.push("position", lx, ly, 0);
    W.push("cuv", (i * wstep + 0.5) / meta.nx, (j * wstep + 0.5) / meta.ny);
  }
  for (let j = 0; j < wy - 1; j++) for (let i = 0; i < wx - 1; i++) {
    const a = j * wx + i; W.idx.push(a, a + 1, a + wx, a + 1, a + wx + 1, a + wx);
  }

  // Земля под зданиями для текстуры рельефа расчёта (канал G).
  const under = new Float32Array(meta.nx * meta.ny);
  for (let k = 0; k < under.length; k++) {
    under[k] = code[k] === meta.building ? ground((k % meta.nx + 0.5) * meta.res, (Math.floor(k / meta.nx) + 0.5) * meta.res) : code[k] / 100;
  }

  const out = { chunks: [], ground: G.pack(), water: W.pack(), under, nBuildings: nb, meta, grid,
    osm: { origin: osm.origin, size: osm.size, kx: osm.kx, ky: osm.ky, buildings: osm.buildings.length } };
  const transfer = [under.buffer, grid];
  for (const [key, { b, s }] of chunks) {
    const [cx, cy] = key.split("_").map(Number);
    const [lx, ly] = toLocal((cx + 0.5) * CHUNK, (cy + 0.5) * CHUNK);
    const item = { key, center: [lx, ly], b: b.n ? b.pack() : null, s: s.n ? s.pack() : null };
    for (const g of [item.b, item.s]) if (g) { transfer.push(g.index.buffer); for (const a of Object.values(g.attrs)) transfer.push(a.buffer); }
    out.chunks.push(item);
  }
  for (const g of [out.ground, out.water]) { transfer.push(g.index.buffer); for (const a of Object.values(g.attrs)) transfer.push(a.buffer); }
  postMessage(out, transfer);
};
