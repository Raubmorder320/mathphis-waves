"""Линейная теория мелкой воды, 1D (docs/theory.md, раздел 2.1).

    η_t + (h·u)_x = 0,    u_t + g·η_x = 0

Это база сравнения: с её решением сверяется каждая нелинейная модель.

Схема — явная «крест» на разнесённой сетке: η хранится в узлах x_j, скорость u —
посередине между узлами и на полшага по времени позже. Второй порядок по x и t,
устойчива при числе Куранта c·Δt/Δx ≤ 1.

Проверки из TASKS.md запускаются командой:  python -m backend.models.linear
"""
import numpy as np

from backend.contract import Params, Result, WaveModel

G = 9.81       # ускорение свободного падения, м/с²
CFL = 0.99     # число Куранта (схема устойчива при значениях до 1)
N_PACKET = 3   # число периодов в пакете для дна slope


# --- Постановка: сетка, дно, начальное условие ------------------------------
# ВРЕМЕННО. По TASKS.md всё это модели берут из backend/common.py. Пока его нет
# в main, постановка задана здесь по описанию из раздела «Общая постановка
# расчёта». Когда common.py появится, функция _setup заменяется вызовами из него.

def _setup(p: Params):
    """Вернуть x, h(x), η₀(x), u₀(x) и признак периодических границ."""
    bottom = getattr(p, "bottom", "flat")
    k = 2 * np.pi / p.wavelength

    if bottom == "flat":
        # Ровное дно, периодические границы, синусоида бежит вправо.
        x = np.arange(p.nx) * (p.L / p.nx)
        h = np.full(p.nx, float(p.depth))
        eta0 = p.a * np.sin(k * x)
        periodic = True
    elif bottom == "slope":
        # Глубина линейно убывает к правому краю, слева стартует пакет, на краях стенки.
        depth_shore = getattr(p, "depth_shore", 1.0)
        x = np.linspace(0.0, p.L, p.nx)
        h = p.depth + (depth_shore - p.depth) * x / p.L
        width = N_PACKET * p.wavelength
        envelope = np.where(x < width, np.sin(np.pi * x / width) ** 2, 0.0)
        eta0 = p.a * envelope * np.sin(k * x)
        periodic = False
    else:
        raise NotImplementedError(
            f"Дно {bottom!r} появится вместе с backend/common.py и данными из data/"
        )

    # Волна, бегущая вправо: u = η·√(g/h). Там, где воды нет, скорость нулевая.
    u0 = np.where(h > 0, eta0 * np.sqrt(G / np.maximum(h, 1e-12)), 0.0)
    return x, h, eta0, u0, periodic


# --- Расчётное ядро: от контракта не зависит --------------------------------

def solve_linear(h, eta0, u0, dx, t_frames, periodic):
    """Решить линейные уравнения мелкой воды и вернуть η в моменты t_frames.

    h, eta0, u0 — глубина, начальное возвышение и скорость в узлах сетки с шагом dx.
    periodic=True — периодические границы (узел nx совпадает с узлом 0).
    periodic=False — вертикальные стенки в первом и последнем узле (u = 0).
    Возвращает массив формы (число кадров, число узлов).
    """
    h = np.asarray(h, dtype=float)
    eta = np.array(eta0, dtype=float)
    u0 = np.asarray(u0, dtype=float)

    # Глубина и скорость на гранях — посередине между соседними узлами.
    if periodic:
        h_face = 0.5 * (h + np.roll(h, -1))
        u = 0.5 * (u0 + np.roll(u0, -1))
    else:
        h_face = 0.5 * (h[:-1] + h[1:])
        u = 0.5 * (u0[:-1] + u0[1:])

    def d_eta(u):
        """−(h·u)_x в узлах."""
        q = h_face * u
        if periodic:
            return -(q - np.roll(q, 1)) / dx
        out = np.empty_like(eta)
        out[1:-1] = -(q[1:] - q[:-1]) / dx
        # У стенки поток через неё нулевой, а ячейка вдвое короче.
        out[0] = -2.0 * q[0] / dx
        out[-1] = 2.0 * q[-1] / dx
        return out

    def d_u(eta):
        """−g·η_x на гранях."""
        if periodic:
            return -G * (np.roll(eta, -1) - eta) / dx
        return -G * (eta[1:] - eta[:-1]) / dx

    # Шаг по времени постоянный, из условия Куранта; t_end попадает ровно на шаг.
    c_max = np.sqrt(G * max(h_face.max(), 1e-12))
    t_end = t_frames[-1]
    n_steps = max(1, int(np.ceil(t_end * c_max / (CFL * dx))))
    dt = t_end / n_steps

    frames = np.empty((len(t_frames), len(eta)))
    frames[0] = eta
    i = 1                    # номер следующего кадра
    eta_prev = None
    # Сдвигаем скорость на полшага вперёд, чтобы схема была второго порядка по времени.
    u = u + 0.5 * dt * d_u(eta)
    for n in range(n_steps):
        eta_next = eta + dt * d_eta(u)
        u = u + dt * d_u(eta_next)
        # Кадры, попавшие между шагами n и n+1, получаем интерполяцией по времени.
        while i < len(t_frames) and t_frames[i] <= (n + 1) * dt * (1 + 1e-12):
            s = t_frames[i] / dt - n          # доля шага, от 0 до 1
            if eta_prev is None:
                frames[i] = eta + s * (eta_next - eta)
            else:
                frames[i] = (eta + 0.5 * s * (eta_next - eta_prev)
                             + 0.5 * s * s * (eta_next - 2 * eta + eta_prev))
            i += 1
        eta_prev, eta = eta, eta_next
    frames[i:] = eta         # на случай ошибки округления в последнем кадре
    return frames


class LinearModel(WaveModel):
    id = "linear"
    name = "Линейная мелкая вода 1D"
    description = (
        "База сравнения для нелинейных моделей. Уравнения η_t + (h·u)_x = 0, u_t + g·η_x = 0: "
        "низкая (a ≪ h) и длинная (λ ≫ h) волна. Бежит со скоростью √(g·h) без искажения формы, "
        "на мелеющем дне растёт по закону Грина. Укручения и обрушения нет."
    )
    dim = 1
    extra_params: list = []

    def solve(self, p: Params) -> Result:
        x, h, eta0, u0, periodic = _setup(p)
        t = np.linspace(0.0, p.t_end, p.n_frames)
        eta = solve_linear(h, eta0, u0, x[1] - x[0], t, periodic)
        return Result(x=x.tolist(), t=t.tolist(), eta=eta.tolist(), h=h.tolist())


# --- Проверки «готово, когда» из TASKS.md -----------------------------------

def _check_full_turn() -> float:
    """Ровное дно: после полного оборота по области профиль совпадает с начальным."""
    p = Params(a=0.5, wavelength=100.0, depth=5.0, L=1000.0, nx=400, n_frames=50)
    p.t_end = p.L / np.sqrt(G * p.depth)
    eta = np.array(LinearModel().solve(p).eta)
    return float(np.max(np.abs(eta[-1] - eta[0])) / p.a)


def _check_green() -> tuple[float, float]:
    """Склон: амплитуда пакета растёт как h^(−1/4). Возвращает (рост, наибольшее отклонение)."""
    p = Params(a=0.1, wavelength=100.0, depth=5.0, L=4000.0, t_end=680.0, nx=1600, n_frames=1361)
    p.bottom = "slope"
    p.depth_shore = 1.0
    r = LinearModel().solve(p)
    x, h, eta = np.array(r.x), np.array(r.h), np.abs(np.array(r.eta))

    # Амплитуда — наибольшее |η| за всё время в данной точке. Берём участок,
    # который пакет прошёл целиком и куда ещё не вернулось отражение от берега.
    amp = eta.max(axis=0)
    zone = (x > 400.0) & (x < 3500.0)
    ref = np.argmax(zone)
    measured = amp[zone] / amp[ref]
    green = (h[ref] / h[zone]) ** 0.25
    return float(green[-1]), float(np.max(np.abs(measured / green - 1.0)))


if __name__ == "__main__":
    err = _check_full_turn()
    print(f"Ровное дно, полный оборот: ошибка {err:.2%} (нужно меньше 1%)")
    growth, dev = _check_green()
    print(f"Склон, закон Грина: амплитуда выросла в {growth:.2f} раза по теории, "
          f"наибольшее отклонение расчёта {dev:.1%} (допуск 10%)")
    assert err < 0.01 and dev < 0.10, "проверка не пройдена"
    print("Обе проверки пройдены")
