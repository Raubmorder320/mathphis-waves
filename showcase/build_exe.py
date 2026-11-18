"""Сборка «Финский залив 3D.exe» одним файлом (PyInstaller).

    pip install pywebview pyinstaller
    python -m showcase.build_exe          # → dist/Финский залив 3D.exe

Внутрь кладутся: модели проекта, данные сценариев, город, дно, кэш тайлов карты
(для работы без интернета) и локальные копии MapLibre и three.js.
"""
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
NAME = "Финский залив 3D"


def make_icon(path):
    """Иконка: волна на тёмно-синем круге."""
    S = 256
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse((8, 8, S - 8, S - 8), fill=(8, 22, 40, 255), outline=(255, 176, 32, 255), width=10)
    for k, (col, dy) in enumerate([((47, 224, 208, 255), 0), ((26, 140, 200, 255), 38), ((14, 80, 140, 255), 76)]):
        pts = [(x, S * 0.42 + dy + 22 * np.sin((x / S) * 2 * np.pi * 1.3 + k)) for x in range(24, S - 23, 4)]
        d.line(pts, fill=col, width=16, joint="curve")
    img.save(path, sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])


def main():
    import PyInstaller.__main__ as pyi

    build = ROOT / "build"
    build.mkdir(exist_ok=True)
    icon = build / "fingulf3d.ico"
    make_icon(icon)
    S = ROOT / "showcase"
    sep = ";"
    datas = [
        (S / "index.html", "showcase"),
        (S / "static", "showcase/static"),
        (S / "data", "showcase/data"),
        (ROOT / "data", "data"),
    ]
    for sub in ("sat", "dem", "labels", "city"):          # кэш карты; сырые EMODnet не нужны
        if (S / ".tiles" / sub).exists():
            datas.append((S / ".tiles" / sub, f"showcase/.tiles/{sub}"))
    args = [
        str(S / "desktop.py"), "--name", "FinGulf3D", "--onefile", "--windowed", "--noconfirm",
        "--icon", str(icon), "--distpath", str(ROOT / "dist"), "--workpath", str(build / "pyi"),
        "--specpath", str(build), "--paths", str(ROOT),
        "--collect-submodules", "uvicorn", "--collect-submodules", "backend", "--collect-submodules", "showcase",
        "--hidden-import", "webview.platforms.edgechromium", "--hidden-import", "clr",
        "--exclude-module", "matplotlib", "--exclude-module", "tkinter", "--exclude-module", "pytest",
    ]
    for src, dst in datas:
        args += ["--add-data", f"{src}{sep}{dst}"]
    pyi.run(args)
    out = ROOT / "dist" / f"{NAME}.exe"
    if out.exists():
        out.unlink()
    shutil.move(str(ROOT / "dist" / "FinGulf3D.exe"), str(out))
    print(f"Готово: {out} ({out.stat().st_size / 1e6:.0f} МБ)")


if __name__ == "__main__":
    sys.exit(main())
