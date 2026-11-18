"""Финский залив 3D как приложение для Windows: свой сервер внутри и окно WebView2.

    python -m showcase.desktop            # запуск без сборки
    python -m showcase.build_exe          # собрать «Финский залив 3D.exe» (PyInstaller)

Сервер FastAPI (showcase/server.py) работает в фоновом потоке на свободном локальном
порту, окно — движок Edge (WebView2), встроенный в Windows 10/11: без Electron и без
браузера. Новые тайлы карты, если появится интернет, кэшируются в %LOCALAPPDATA%/FinGulf3D.
"""
import multiprocessing
import os
import socket
import sys
import threading
import time
from pathlib import Path


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main():
    multiprocessing.freeze_support()          # процессы расчёта разреза в собранном .exe
    if getattr(sys, "frozen", False):
        # Без консоли stdout/stderr = None — логирование uvicorn не должно о них спотыкаться.
        devnull = open(os.devnull, "w")
        sys.stdout = sys.stdout or devnull
        sys.stderr = sys.stderr or devnull
        cache = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "FinGulf3D" / "tiles"
        os.environ.setdefault("SHOWCASE_CACHE", str(cache))

    import uvicorn
    import webview

    if os.environ.get("FINGULF_DEBUG_PORT"):             # отладка WebView2 по протоколу DevTools
        webview.settings["REMOTE_DEBUGGING_PORT"] = int(os.environ["FINGULF_DEBUG_PORT"])

    from showcase.server import app

    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_config=None,
                                           access_log=False, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(300):                        # до 30 с: первый запуск распаковывает .exe
        if server.started:
            break
        time.sleep(0.1)

    webview.create_window("Финский залив 3D", f"http://127.0.0.1:{port}/", width=1600, height=950,
                          min_size=(1000, 650), background_color="#05080D", maximized=True)
    webview.start(gui="edgechromium", private_mode=False,
                  storage_path=str(Path(os.environ.get("LOCALAPPDATA", Path.home())) / "FinGulf3D" / "webview"))
    server.should_exit = True


if __name__ == "__main__":
    main()
