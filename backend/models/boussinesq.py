"""Уравнения Буссинеска в форме Перегрина, 1D (docs/theory.md, раздел 2.3).

    η_t + [(h + η)·u]_x = 0
    u_t + u·u_x + g·η_x = (h/2)·(h·u_t)_xx − (h²/6)·u_xxt

К Сен-Венану добавлена первая поправка на негидростатическое давление: короткие
волны бегут медленнее длинных, ω² = g·h·k² / (1 + (kh)²/3). Поэтому фронт не
рвётся, а распадается на цепочку волн (ондулярный бор). Дно может быть переменным.

Схема. Сетка узловая, производные η_x, u_x, ((h+η)·u)_x — центральные разности
4-го порядка. Дисперсионный член неявный: уравнение импульса переписано для
w = u_t в виде
    w − (h/2)·(h·w)_xx + (h²/6)·w_xx = −u·u_x − g·η_x + (обрушение),
на каждом шаге это трёхдиагональная система относительно w (scipy.linalg.solve_banded;
при периодических границах — циклическая, через формулу Шермана — Моррисона).
По времени — Рунге — Кутта 4-го порядка, шаг из условия Куранта. После каждого
шага слабый фильтр Шапиро гасит пилу с шагом в один узел (см. _filter).

Обрушение (раздел 4.2). Модель Буссинеска сама не обрушивается: волна у берега
растёт, пока расчёт не развалится. Поэтому там, где высота волны H (размах η
в окне длиной λ) превышает γ·h, γ = 0,78, включается локальная искусственная
вязкость на переднем склоне волны, по Kennedy et al. (2000):
    ν = B·δ²·h·η_t,   δ = 1,2,   B растёт от 0 до 1, пока H/h растёт от γ до 1,2·γ.
Она гасит энергию на фронте так же, как бор в модели Сен-Венана.

Накат на сухой берег модель не описывает: берег — стенка там, где глубина
становится меньше амплитуды a (но не меньше H_WALL), дальше η = 0. Мельче волна
уже обрушилась бы, и уравнения Буссинеска там неприменимы.

Проверки из TASKS.md запускаются командой:  python -m backend.models.boussinesq
"""
import numpy as np
from scipy.linalg import solve_banded
from scipy.ndimage import maximum_filter1d, minimum_filter1d

from backend.common import G, frame_times, setup, u_simple_wave
from backend.contract import Params, Result, WaveModel

CFL = 0.5          # число Куранта
H_WALL = 0.25      # наименьшая глубина у берега-стенки, м (стенка — где h < max(a, H_WALL))
GAMMA = 0.78       # критерий обрушения H/h (Мак-Кауэн)
DELTA_B = 1.2      # масштаб вязкости обрушения в долях глубины (Kennedy et al., 2000)
SIGMA = 0.05       # сила фильтра сеточного шума за один шаг (см. _filter)


# --- Расчётное ядро: от контракта не зависит --------------------------------

def _d1(f, dx, periodic, odd):
    """Первая производная центральной разностью 4-го порядка.

    У стенки значения продолжаются зеркально: чётно для η и h, нечётно для u
    (на стенке u = 0).
    """
    if periodic:
        e = np.pad(f, 2, mode="wrap")
    else:
        e = np.pad(f, 2, mode="reflect")
        if odd:
            e[:2] = -e[:2]
            e[-2:] = -e[-2:]
    return (e[:-4] - 8 * e[1:-3] + 8 * e[3:-1] - e[4:]) / (12 * dx)


def _filter(f, periodic, odd):
    """Слабый фильтр Шапиро 4-го порядка: f − (σ/16)·δ⁴f.

    Центральные разности ничем не гасят пилу с шагом в один узел, и у берега
    она нарастает, пока не развалит расчёт. Фильтр убирает за шаг долю σ такой
    пилы, а волну, на длину которой приходится N узлов, — только долю
    σ·sin⁴(π/N): при N = 40 это 2·10⁻⁶ за шаг.
    """
    if periodic:
        e = np.pad(f, 2, mode="wrap")
    else:
        e = np.pad(f, 2, mode="reflect")
        if odd:
            e[:2] = -e[:2]
            e[-2:] = -e[-2:]
    d4 = e[:-4] - 4 * e[1:-3] + 6 * e[2:-2] - 4 * e[3:-1] + e[4:]
    out = f - (SIGMA / 16) * d4
    if not periodic and odd:
        out[0] = out[-1] = 0.0
    return out


def _dispersion_matrix(h, dx):
    """Диагонали оператора w − (h/2)·(h·w)_xx + (h²/6)·w_xx (разности 2-го порядка).

    Возвращает (нижняя, главная, верхняя): нижняя[j] — коэффициент при w[j−1]
    в уравнении j, верхняя[j] — при w[j+1]. Соседи по кругу (для периодики) — np.roll.
    """
    r = 1.0 / dx ** 2
    lower = r * (h * h / 6 - 0.5 * h * np.roll(h, 1))
    upper = r * (h * h / 6 - 0.5 * h * np.roll(h, -1))
    main = 1.0 + r * (2.0 / 3.0) * h * h
    return lower, main, upper


def _make_solver(h, dx, periodic):
    """Функция, решающая трёхдиагональную систему M·w = F для дисперсионного оператора."""
    lower, main, upper = _dispersion_matrix(h, dx)
    n = len(h)

    if not periodic:
        # Стенки: на краях u = 0 всё время, значит и w = u_t = 0. Неизвестные — внутренние узлы.
        ab = np.zeros((3, n - 2))
        ab[0, 1:] = upper[1:-2]
        ab[1] = main[1:-1]
        ab[2, :-1] = lower[2:-1]

        def solve(F):
            w = np.zeros(n)
            w[1:-1] = solve_banded((1, 1), ab, F[1:-1])
            return w
        return solve

    # Периодика: циклическая система. Угловые элементы убираем поправкой ранга 1
    # (Шерман — Моррисон): M = T + v·wᵀ, где T трёхдиагональная; решаем две системы с T.
    corner_top, corner_bottom = lower[0], upper[-1]      # M[0, n−1] и M[n−1, 0]
    gamma = -main[0]
    d = main.copy()
    d[0] -= gamma
    d[-1] -= corner_bottom * corner_top / gamma
    ab = np.zeros((3, n))
    ab[0, 1:] = upper[:-1]
    ab[1] = d
    ab[2, :-1] = lower[1:]
    v = np.zeros(n)
    v[0], v[-1] = gamma, corner_bottom
    z = solve_banded((1, 1), ab, v)
    denom = 1.0 + z[0] + corner_top * z[-1] / gamma

    def solve(F):
        y = solve_banded((1, 1), ab, F)
        return y - z * (y[0] + corner_top * y[-1] / gamma) / denom
    return solve


def solve_boussinesq(h, eta0, u0, dx, t_frames, periodic, window, gamma=GAMMA):
    """Решить уравнения Перегрина и вернуть η в моменты t_frames.

    h, eta0, u0 — глубина, начальное возвышение и скорость в узлах с шагом dx.
    periodic=True — периодические границы, False — стенки (u = 0) на краях.
    window — длина окна (м), в котором высота волны сравнивается с глубиной;
    gamma — порог обрушения H/h, при gamma ≤ 0 обрушение выключено.
    Возвращает массив (число кадров, число узлов).
    """
    h = np.asarray(h, dtype=float)
    eta = np.array(eta0, dtype=float)
    u = np.array(u0, dtype=float)
    if not periodic:
        u[0] = u[-1] = 0.0
    solve_w = _make_solver(h, dx, periodic)
    breaking = gamma > 0
    size = max(3, int(round(window / dx)) | 1)            # нечётное число узлов окна
    mode = "wrap" if periodic else "nearest"

    def viscosity(eta, eta_t):
        """Коэффициент вязкости обрушения ν в узлах."""
        height = (maximum_filter1d(eta, size, mode=mode)
                  - minimum_filter1d(eta, size, mode=mode))
        ratio = height / h
        b = np.clip((ratio / gamma - 1.0) / 0.2, 0.0, 1.0)
        return b * DELTA_B ** 2 * h * np.maximum(eta_t, 0.0)

    def rhs(eta, u):
        H = h + eta
        if np.any(H <= 0):
            raise RuntimeError("Волна осушила дно: модель Буссинеска наката не описывает")
        eta_t = -_d1(H * u, dx, periodic, odd=True)
        F = -u * _d1(u, dx, periodic, odd=True) - G * _d1(eta, dx, periodic, odd=False)
        nu = None
        if breaking:
            nu = viscosity(eta, eta_t)
            if nu.any():
                # (1/H)·(ν·(H·u)_x)_x — разности 2-го порядка через грани.
                q = H * u
                qx = (np.roll(q, -1) - q) / dx
                nu_f = 0.5 * (nu + np.roll(nu, -1))
                flux = nu_f * qx
                if not periodic:
                    flux[-1] = 0.0                       # грань за стенкой
                div = (flux - np.roll(flux, 1)) / dx
                if not periodic:
                    div[0] = 0.0
                F = F + div / H
        w = solve_w(F)
        return eta_t, w, nu

    frames = np.empty((len(t_frames), len(eta)))
    frames[0] = eta
    t = float(t_frames[0])
    for j in range(1, len(t_frames)):
        while t < t_frames[j] - 1e-12:
            k1e, k1u, nu = rhs(eta, u)
            speed = np.max(np.abs(u) + np.sqrt(G * (h + np.maximum(eta, 0.0))))
            dt = CFL * dx / max(speed, 1e-12)
            if nu is not None and nu.max() > 0:
                dt = min(dt, 0.25 * dx * dx / nu.max())  # устойчивость явной вязкости
            dt = min(dt, t_frames[j] - t)
            k2e, k2u, _ = rhs(eta + 0.5 * dt * k1e, u + 0.5 * dt * k1u)
            k3e, k3u, _ = rhs(eta + 0.5 * dt * k2e, u + 0.5 * dt * k2u)
            k4e, k4u, _ = rhs(eta + dt * k3e, u + dt * k3u)
            eta = _filter(eta + dt / 6 * (k1e + 2 * k2e + 2 * k3e + k4e), periodic, odd=False)
            u = _filter(u + dt / 6 * (k1u + 2 * k2u + 2 * k3u + k4u), periodic, odd=True)
            t += dt
            if not (np.all(np.isfinite(eta)) and np.all(np.isfinite(u))):
                raise RuntimeError(f"Расчёт Буссинеска разошёлся при t = {t:.2f} с")
        frames[j] = eta
    return frames


class BoussinesqModel(WaveModel):
    id = "boussinesq"
    name = "Буссинеск (Перегрин) 1D"
    description = (
        "Слабая нелинейность и слабая дисперсия при переменном дне, волны в обе стороны: "
        "η_t + ((h+η)·u)_x = 0, u_t + u·u_x + g·η_x = (h/2)·(h·u_t)_xx − (h²/6)·u_xxt. "
        "Короткие волны бегут медленнее длинных, поэтому вместо разрыва за фронтом "
        "получается цепочка волн (ондулярный бор). Обрушение — по критерию H/h > γ. "
        "Наката на сухой берег нет: берег — стенка."
    )
    dim = 1
    extra_params = [
        {"name": "gamma", "label": "Порог обрушения H/h (0 — выключить)", "unit": "",
         "default": GAMMA, "min": 0.0, "max": 1.2},
    ]

    def solve(self, p: Params) -> Result:
        s = setup(p)
        t = frame_times(p)
        gamma = float(self.extras(p)["gamma"])
        # Берег — стенка в последнем узле, где глубина ещё не меньше амплитуды.
        h_wall = max(H_WALL, abs(p.a))
        shallow = np.flatnonzero(s.h < h_wall)
        wet = int(shallow[0]) if len(shallow) else len(s.x)
        if wet < 5:
            raise ValueError(f"У левого края глубина меньше {h_wall:g} м: увеличьте depth")
        h, eta0 = s.h[:wet], s.eta0[:wet]
        eta = np.zeros((len(t), len(s.x)))
        eta[:, :wet] = solve_boussinesq(h, eta0, u_simple_wave(h, eta0), s.dx, t,
                                        s.periodic, window=p.wavelength, gamma=gamma)
        return Result(x=s.x.tolist(), t=t.tolist(), eta=eta.tolist(), h=s.h.tolist())


# --- Проверки «готово, когда» из TASKS.md -----------------------------------

def _check_phase_speed() -> list[tuple[float, float, float]]:
    """Малая волна на ровном дне при разных kh: скорость в расчёте против формулы.

    Возвращает список (kh, c/c₀ в расчёте, c/c₀ по формуле ω² = ghk²/(1 + (kh)²/3)).
    """
    out = []
    h, lam = 5.0, 100.0
    c0 = np.sqrt(G * h)
    for kh in (0.3, 1.0, 2.0):
        lam = 2 * np.pi * h / kh
        n_per, n_waves = 40, 4                          # узлов на длину волны, волн в области
        L, n = n_waves * lam, n_waves * n_per
        x = np.arange(n) * (L / n)
        k = 2 * np.pi / lam
        c_th = c0 / np.sqrt(1 + kh ** 2 / 3)
        eta0 = 1e-4 * np.sin(k * x)
        u0 = eta0 * c_th / h                            # бегущая вправо линейная волна
        T = L / c_th                                    # за это время волна обходит область
        eta = solve_boussinesq(np.full(n, h), eta0, u0, L / n, np.array([0.0, T]),
                               periodic=True, window=lam, gamma=0.0)
        f0, f1 = np.fft.rfft(eta[0])[n_waves], np.fft.rfft(eta[-1])[n_waves]
        lag = np.angle(f1 / f0) / k                     # отставание от теории за время T, м
        c_num = c_th - lag / T
        out.append((kh, c_num / c0, c_th / c0))
    return out


def _check_soliton() -> tuple[float, float]:
    """Солитон из раздела 2.4 (A = 0,5 м, h = 20 м) проходит всю область.

    Солитон КдФ — не точное решение уравнений Перегрина, поэтому сравниваем форму
    после совмещения гребней. Возвращает (изменение амплитуды, ошибку формы) в долях A.
    """
    A, h, L, n = 0.5, 20.0, 4000.0, 1600
    x = np.arange(n) * (L / n)
    c0 = np.sqrt(G * h)
    V, delta = c0 * (1 + A / (2 * h)), np.sqrt(4 * h ** 3 / (3 * A))

    def sol(xc):
        return sum(A / np.cosh((x - xc - m * L) / delta) ** 2 for m in (-1, 0, 1))

    eta0 = sol(L / 2)
    u0 = V * eta0 / (h + eta0)                         # расход стационарной волны
    eta = solve_boussinesq(np.full(n, h), eta0, u0, L / n, np.array([0.0, L / V]),
                           periodic=True, window=2 * delta, gamma=0.0)[-1]
    shift = np.argmax(eta) - np.argmax(eta0)
    aligned = np.roll(eta, -shift)
    return float(eta.max() / A - 1), float(np.abs(aligned - eta0).max() / A)


def _crests_per_wavelength(eta, wavelength, L):
    """Среднее число гребней (локальных максимумов выше нуля) на одну длину волны."""
    peaks = (eta > np.roll(eta, 1)) & (eta >= np.roll(eta, -1)) & (eta > 0)
    return peaks.sum() / (L / wavelength)


def _check_undular_bore() -> tuple[float, float, float]:
    """Значения по умолчанию (как у saint_venant, где при t ≈ 15 с появляется разрыв).

    Через 60 с у Сен-Венана на каждой длине волны один гребень (пила), а у Буссинеска
    за фронтом цепочка волн. Возвращает (гребней на λ у Буссинеска,
    гребней на λ у Сен-Венана, наибольшую высоту гребня в долях a).
    """
    from backend.models.saint_venant import SaintVenantModel
    p = Params(nx=2000, n_frames=2)                   # a = 0,5 м, λ = 100 м, h = 5 м, 60 с
    eta_b = np.array(BoussinesqModel().solve(p).eta)[-1]
    eta_sv = np.array(SaintVenantModel().solve(p).eta)[-1]
    return (float(_crests_per_wavelength(eta_b, p.wavelength, p.L)),
            float(_crests_per_wavelength(eta_sv, p.wavelength, p.L)),
            float(eta_b.max() / p.a))


def _check_breaking() -> tuple[float, float, float]:
    """Пакет a = 0,5 м на склоне 5 м → 0,5 м: с обрушением расчёт доходит до конца.

    Возвращает (наибольшее H/h без учёта первых и последних узлов с обрушением,
    наибольшее H/h без обрушения или nan, если расчёт без обрушения развалился, долю
    сохранённой энергии с обрушением).
    """
    p = Params(a=0.5, wavelength=100.0, depth=5.0, L=1000.0, t_end=120.0, nx=1000, n_frames=241)
    p.bottom, p.depth_shore = "slope", 0.5
    m = BoussinesqModel()

    def max_ratio(eta, h):
        wet = h >= max(H_WALL, p.a)
        size = int(round(p.wavelength / (p.L / p.nx))) | 1
        height = np.array([maximum_filter1d(e, size, mode="nearest")
                           - minimum_filter1d(e, size, mode="nearest") for e in eta])
        return float((height[:, wet] / h[wet]).max())

    r = m.solve(p)
    eta, h = np.array(r.eta), np.array(r.h)
    energy = (eta ** 2).sum(axis=1)
    with_breaking = max_ratio(eta, h)
    p.extra = {"gamma": 0.0}
    try:
        r0 = m.solve(p)
        without = max_ratio(np.array(r0.eta), np.array(r0.h))
    except RuntimeError:
        without = float("nan")
    return with_breaking, without, float(energy[-1] / energy.max())


if __name__ == "__main__":
    worst = 0.0
    for kh, c_num, c_th in _check_phase_speed():
        err = c_num / c_th - 1
        worst = max(worst, abs(err))
        print(f"Фазовая скорость при kh = {kh}: расчёт {c_num:.4f}·c₀, формула {c_th:.4f}·c₀, "
              f"отличие {err:+.2%} (допуск 2%)")
    da, ds = _check_soliton()
    print(f"Солитон после прохода всей области: амплитуда {da:+.1%}, ошибка формы {ds:.1%} от A")
    nb, nsv, top = _check_undular_bore()
    print(f"Значения по умолчанию через 60 с: гребней на длину волны — Буссинеск {nb:.1f}, "
          f"Сен-Венан {nsv:.1f}; высший гребень {top:.2f}·a")
    rb, r0, e_left = _check_breaking()
    r0_text = "расчёт развалился" if np.isnan(r0) else f"H/h до {r0:.2f}"
    print(f"Склон к мелководью: с обрушением H/h до {rb:.2f}, осталось {e_left:.0%} энергии; "
          f"без обрушения {r0_text}")
    assert worst < 0.02, "фазовая скорость"
    assert abs(da) < 0.05 and ds < 0.05, "солитон"
    assert nb >= 2 and nsv < 1.5, "ондулярный бор"
    assert rb < 1.2 * GAMMA * 1.2, "обрушение"
    print("Все проверки пройдены")
