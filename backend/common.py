"""Общая постановка расчёта: сетка, дно и начальное условие.

Все одномерные модели берут постановку отсюда, чтобы считать одну и ту же задачу:
иначе их решения нельзя сравнивать. Варианты дна (Params.bottom):

- flat  — ровное дно, периодические границы, η₀ = a·sin(2πx/λ);
- slope — глубина линейно убывает от depth до depth_shore, слева пакет из N_PACKET
          периодов, на краях стенки; при depth_shore ≤ 0 у правого края сухой берег;
- gulf  — разрез залива от открытой части до берега Невской губы (data/gulf_profile.npz),
          пакет и стенки как у slope.

Начальная скорость зависит от модели: волна, бегущая вправо, в линейной и нелинейной
теории задаётся по-разному. Для этого две функции: u_linear и u_simple_wave.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from backend.contract import Params

G = 9.81       # ускорение свободного падения, м/с²
N_PACKET = 3   # число периодов в пакете для дна slope и gulf
GULF_PROFILE = Path(__file__).resolve().parent.parent / "data" / "gulf_profile.npz"


@dataclass
class Setup:
    x: np.ndarray      # узлы сетки, м
    h: np.ndarray      # глубина в узлах, м; на суше 0 и меньше
    eta0: np.ndarray   # начальное возвышение, м
    periodic: bool     # True — периодические границы, False — стенки на краях

    @property
    def dx(self) -> float:
        return float(self.x[1] - self.x[0])


def load_gulf_profile() -> tuple[np.ndarray, np.ndarray]:
    """Разрез залива: x (м, от западного края) и h (м, суша — 0 и меньше)."""
    if not GULF_PROFILE.exists():
        raise NotImplementedError("Нет data/gulf_profile.npz — данные из ветки vlad/analysis")
    with np.load(GULF_PROFILE) as d:
        return np.asarray(d["x"], float), np.asarray(d["h"], float)


def setup(p: Params) -> Setup:
    """Сетка, дно и начальное возвышение по параметрам p."""
    k = 2 * np.pi / p.wavelength

    if p.bottom == "flat":
        # Периодическая сетка: узел nx совпадал бы с узлом 0.
        x = np.arange(p.nx) * (p.L / p.nx)
        h = np.full(p.nx, float(p.depth))
        return Setup(x, h, p.a * np.sin(k * x), periodic=True)

    if p.bottom == "slope":
        x = np.linspace(0.0, p.L, p.nx)
        h = p.depth + (p.depth_shore - p.depth) * x / p.L
    elif p.bottom == "gulf":
        xp, hp = load_gulf_profile()
        x = np.linspace(xp.min(), xp.max(), p.nx)
        h = np.interp(x, xp, hp)
    else:
        raise ValueError(f"Неизвестное дно {p.bottom!r}: нужно flat, slope или gulf")

    # Слева стартует пакет из нескольких периодов с гладкой огибающей.
    s = x - x[0]
    width = N_PACKET * p.wavelength
    envelope = np.where(s < width, np.sin(np.pi * s / width) ** 2, 0.0)
    eta0 = np.where(h > 0, p.a * envelope * np.sin(k * s), 0.0)
    return Setup(x, h, eta0, periodic=False)


def u_linear(h: np.ndarray, eta: np.ndarray) -> np.ndarray:
    """Скорость линейной волны, бегущей вправо: u = η·√(g/h). На суше 0."""
    return np.where(h > 0, eta * np.sqrt(G / np.maximum(h, 1e-12)), 0.0)


def u_simple_wave(h: np.ndarray, eta: np.ndarray) -> np.ndarray:
    """Скорость простой нелинейной волны, бегущей вправо: u = 2(√(gH) − √(gh)).

    Инвариант Римана u − 2√(gH) тот же, что у покоящейся воды (docs/theory.md, 2.2).
    """
    H = np.maximum(h + eta, 0.0)
    return np.where(h > 0, 2 * (np.sqrt(G * H) - np.sqrt(G * np.maximum(h, 0.0))), 0.0)


def frame_times(p: Params) -> np.ndarray:
    """Моменты кадров: n_frames штук от 0 до t_end."""
    return np.linspace(0.0, p.t_end, p.n_frames)
