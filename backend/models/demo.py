import numpy as np

from backend.contract import Params, Result, WaveModel

G = 9.81  # ускорение свободного падения, м/с²


class DemoModel(WaveModel):
    id = "demo"
    name = "Демо: бегущая синусоида"
    description = (
        "Демо для проверки интерфейса, настоящие модели пишет Ваня. "
        "η = a·sin(k(x − c·t)), c = √(g·h), дно ровное."
    )

    def solve(self, p: Params) -> Result:
        x = np.linspace(0, p.L, p.nx)
        t = np.linspace(0, p.t_end, p.n_frames)
        k = 2 * np.pi / p.wavelength
        c = np.sqrt(G * p.depth)
        eta = p.a * np.sin(k * (x[None, :] - c * t[:, None]))
        h = np.full_like(x, p.depth)
        return Result(x=x.tolist(), t=t.tolist(), eta=eta.tolist(), h=h.tolist())
