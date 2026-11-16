import numpy as np

from backend.common import G, frame_times, setup
from backend.contract import Params, Result, WaveModel


class DemoModel(WaveModel):
    id = "demo"
    name = "Демо: перенос без искажения"
    description = (
        "Заглушка для проверки страницы, не физика. Начальный профиль из общей постановки "
        "просто сдвигается вправо со скоростью √(g·depth)."
    )

    def solve(self, p: Params) -> Result:
        s = setup(p)
        t = frame_times(p)
        c = np.sqrt(G * p.depth)
        shifted = s.x[None, :] - c * t[:, None]
        if s.periodic:
            eta = np.array([np.interp(row, s.x, s.eta0, period=p.L) for row in shifted])
        else:
            eta = np.array([np.interp(row, s.x, s.eta0, left=0.0, right=0.0) for row in shifted])
        eta = np.where(s.h[None, :] > 0, eta, 0.0)
        return Result(x=s.x.tolist(), t=t.tolist(), eta=eta.tolist(), h=s.h.tolist())
