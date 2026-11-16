"""Уравнение Кортевега — де Фриза (docs/theory.md, раздел 2.4).

    η_t + c₀·η_x + α·η·η_x + β·η_xxx = 0,    α = 3c₀/(2h),  β = c₀·h²/6

Волна бежит только вправо. Нелинейность (α) укручает фронт, дисперсия (β)
расплывает его; их баланс даёт солитоны.

Схема — псевдоспектральная на периодической сетке: производные по x через БПФ,
по времени ETDRK4 (Cox, Matthews 2002; Kassam, Trefethen 2005). Линейная часть
c₀·η_x + β·η_xxx интегрируется точно, поэтому шаг ограничен только нелинейным
членом, а не жёсткой дисперсией коротких волн. Нелинейный член очищается от
наложения частот по правилу 2/3. Уравнение выведено для постоянной глубины,
поэтому модель считает только ровное дно.

Проверки из TASKS.md запускаются командой:  python -m backend.models.kdv
"""
import numpy as np

from backend.common import G, frame_times, setup
from backend.contract import Params, Result, WaveModel

CFL = 0.1    # α·max|η|·k_max·Δt за шаг; при 0,1 энергия сохраняется до 10⁻⁷
N_CONTOUR = 32  # точек на контуре для коэффициентов ETDRK4


def _etd_coefficients(lin, dt):
    """Коэффициенты ETDRK4 для линейного оператора lin (диагонального в Фурье) и шага dt.

    Функции вида (e^z − 1)/z считаются средним по окружности вокруг z, иначе при
    малых |z| они теряют точность из-за вычитания близких чисел.
    """
    r = np.exp(2j * np.pi * (np.arange(1, N_CONTOUR + 1) - 0.5) / N_CONTOUR)  # вся окружность: lin мнимый
    z = dt * lin[:, None] + r[None, :]
    ez = np.exp(z)
    q = dt * np.mean((np.exp(z / 2) - 1) / z, axis=1)
    f1 = dt * np.mean((-4 - z + ez * (4 - 3 * z + z * z)) / z ** 3, axis=1)
    f2 = dt * np.mean((2 + z + ez * (z - 2)) / z ** 3, axis=1)
    f3 = dt * np.mean((-4 - 3 * z - z * z + ez * (4 - z)) / z ** 3, axis=1)
    return np.exp(dt * lin), np.exp(dt * lin / 2), q, f1, f2, f3


def solve_kdv(eta0, h, L, t_frames):
    """Решить КдФ на периодической области длины L и вернуть η в моменты t_frames.

    eta0 — начальный профиль в n равноотстоящих узлах x_j = j·L/n, h — глубина.
    Возвращает массив (число кадров, n).
    """
    eta0 = np.asarray(eta0, dtype=float)
    n = len(eta0)
    c0 = np.sqrt(G * h)
    alpha, beta = 1.5 * c0 / h, c0 * h * h / 6

    k = 2 * np.pi * np.fft.rfftfreq(n, d=L / n)
    lin = 1j * (beta * k ** 3 - c0 * k)               # η̂_t = lin·η̂ + N(η̂)
    g = -0.5j * alpha * k * (k < (2 / 3) * k.max() * (1 + 1e-12))  # правило 2/3

    def nonlinear(v):
        """N(η̂) = −α·(η·η_x)^ = −(iαk/2)·(η²)^."""
        eta = np.fft.irfft(v, n)
        return g * np.fft.rfft(eta * eta)

    frames = np.empty((len(t_frames), n))
    frames[0] = eta0
    v = np.fft.rfft(eta0)
    k_max = (2 / 3) * k.max()
    for j in range(1, len(t_frames)):
        # Между кадрами шаг постоянный, чтобы коэффициенты считались один раз на кадр.
        span = t_frames[j] - t_frames[j - 1]
        eta_max = max(np.abs(np.fft.irfft(v, n)).max(), 1e-12)
        steps = max(1, int(np.ceil(span * alpha * eta_max * k_max / CFL)))
        E, E2, q, f1, f2, f3 = _etd_coefficients(lin, span / steps)
        for _ in range(steps):
            nv = nonlinear(v)
            a = E2 * v + q * nv
            na = nonlinear(a)
            b = E2 * v + q * na
            nb = nonlinear(b)
            c = E2 * a + q * (2 * nb - nv)
            v = E * v + f1 * nv + 2 * f2 * (na + nb) + f3 * nonlinear(c)
        eta = np.fft.irfft(v, n)
        if not np.all(np.isfinite(eta)):
            raise RuntimeError(f"Расчёт КдФ разошёлся при t = {t_frames[j]:.2f} с")
        frames[j] = eta
    return frames


class KdVModel(WaveModel):
    id = "kdv"
    name = "КдФ (Кортевег — де Фриз)"
    description = (
        "Слабая нелинейность и слабая дисперсия, волна бежит в одну сторону: "
        "η_t + c₀·η_x + (3c₀/2h)·η·η_x + (c₀h²/6)·η_xxx = 0. Фронт укручается, но вместо "
        "обрушения волна распадается на цепочку солитонов. Применима при числе Урселла порядка 1–30. "
        "Только ровное дно."
    )

    def solve(self, p: Params) -> Result:
        if p.bottom != "flat":
            raise NotImplementedError(
                "КдФ считается только на ровном дне (bottom = flat): уравнение выведено "
                "для постоянной глубины. Для склона используйте Буссинеска или Сен-Венана."
            )
        s = setup(p)
        t = frame_times(p)
        eta = solve_kdv(s.eta0, p.depth, p.L, t)
        return Result(x=s.x.tolist(), t=t.tolist(), eta=eta.tolist(), h=s.h.tolist())


# --- Проверки «готово, когда» из TASKS.md -----------------------------------

def _soliton(x, t, A, h, L):
    """Точный солитон η = A·sech²((x − Vt)/Δ) на периодической области (с ближайшими копиями)."""
    c0 = np.sqrt(G * h)
    V, delta = c0 * (1 + A / (2 * h)), np.sqrt(4 * h ** 3 / (3 * A))
    xc = L / 4 + V * t
    return sum(A / np.cosh((x - xc - m * L) / delta) ** 2 for m in (-2, -1, 0, 1, 2))


def _check_soliton() -> float:
    """Солитон A = 0,5 м на глубине 20 м проходит всю область; ошибка формы к амплитуде."""
    A, h, L, n = 0.5, 20.0, 4000.0, 1024
    x = np.arange(n) * (L / n)
    V = np.sqrt(G * h) * (1 + A / (2 * h))
    t = np.array([0.0, L / V])
    eta = solve_kdv(_soliton(x, 0.0, A, h, L), h, L, t)
    return float(np.abs(eta[-1] - _soliton(x, t[-1], A, h, L)).max() / A)


def _check_invariants() -> tuple[float, float]:
    """Значения по умолчанию: относительное изменение ∫η² и ∫η за t_end (∫η сравнивается с a·L)."""
    p = Params()
    s = setup(p)
    eta = solve_kdv(s.eta0, p.depth, p.L, frame_times(p))
    mass = eta.sum(axis=1) * s.dx
    energy = (eta ** 2).sum(axis=1) * s.dx
    return float(abs(mass[-1] - mass[0]) / (p.a * p.L)), float(abs(energy[-1] / energy[0] - 1))


def _check_fission() -> tuple[int, float]:
    """Широкий горб (a = 0,5 м, ширина 2 км, h = 20 м, Ur ≈ 250) за 6 часов распадается на солитоны.

    Число солитонов из горба η₀ = a·sin²(πx/w) по теории обратной задачи рассеяния
    N ≈ 1 + (2w/π²)·√(3a/(2h³)) ≈ 1 + 0,25·√Ur (docs/theory.md, 2.4): при Ur ≈ 16 это
    порог второго солитона, поэтому горб шириной 500 м даёт один солитон. Здесь N ≈ 4.
    Возвращает число гребней выше a/4 и высоту ведущего гребня в долях a.
    """
    a, w, h, L, n = 0.5, 2000.0, 20.0, 16000.0, 2048
    x = np.arange(n) * (L / n)
    eta0 = np.where(x < w, a * np.sin(np.pi * x / w) ** 2, 0.0)
    eta = solve_kdv(eta0, h, L, np.array([0.0, 6 * 3600.0]))[-1]
    peaks = (eta[1:-1] > eta[:-2]) & (eta[1:-1] > eta[2:]) & (eta[1:-1] > a / 4)
    return int(peaks.sum()), float(eta.max() / a)


if __name__ == "__main__":
    err = _check_soliton()
    print(f"Солитон после прохода всей области: ошибка формы {err:.2%} (нужно меньше 1%)")
    dm, de = _check_invariants()
    print(f"Значения по умолчанию: масса изменилась на {dm:.1e}, энергия на {de:.1e} (нужно меньше 1e-6)")
    n_peaks, lead = _check_fission()
    print(f"Горб с Ur ≈ 250 через 6 часов: {n_peaks} гребня выше a/4, ведущий выше начального в {lead:.2f} раза")
    assert err < 0.01, "солитон"
    assert dm < 1e-6 and de < 1e-6, "инварианты"
    assert n_peaks >= 3 and lead > 1.5, "распад на солитоны"
    print("Все проверки пройдены")
