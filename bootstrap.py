# SPDX-License-Identifier: MIT
# Copyright (c) 2024-2025 VueOSD — https://github.com/wkumik/Digital-FPV-OSD-Tool
import sys, os, subprocess, threading, shutil, traceback, time

HERE    = os.path.dirname(os.path.abspath(__file__))
PYTHON  = sys.executable
SFPATH  = os.path.join(os.environ.get("TEMP", HERE), "vueosd_splash.txt")


def _show_error(msg):
    try:
        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk(); root.withdraw()
        messagebox.showerror("VueOSD — Error", msg)
        root.destroy()
    except Exception:
        pass


def _hta_step(prog, msg):
    """Update the HTA splash status file (used before PyQt6 is ready)."""
    try:
        with open(SFPATH, "w") as f:
            f.write(f"{prog}\n{msg}")
    except Exception:
        pass


def _hta_close():
    """Tell the HTA to close itself."""
    try:
        with open(SFPATH, "w") as f:
            f.write("CLOSE")
    except Exception:
        pass


# ── Relaunch as pythonw.exe to kill the console window ────────────────────────
if sys.platform == "win32" and "pythonw" not in PYTHON.lower():
    pythonw = os.path.join(os.path.dirname(PYTHON), "pythonw.exe")
    if os.path.exists(pythonw):
        subprocess.Popen([pythonw, os.path.abspath(__file__)] + sys.argv[1:],
                         cwd=HERE)
        sys.exit(0)
    # pythonw not found — continue under python.exe (console visible but functional)


def _pip(*packages):
    subprocess.run([PYTHON, "-m", "pip", "install", "--user", "--quiet",
                    *packages], check=False)


def _has(*packages):
    try:
        r = subprocess.run([PYTHON, "-c", "import " + ",".join(packages)],
                           capture_output=True, timeout=10)
        return r.returncode == 0
    except Exception:
        return False


def _ffmpeg_ok():
    return shutil.which("ffmpeg") is not None


def _refresh_path_from_registry():
    if sys.platform != "win32":
        return
    try:
        import winreg
        parts = []
        for hive, key in [
            (winreg.HKEY_LOCAL_MACHINE,
             r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"),
            (winreg.HKEY_CURRENT_USER, r"Environment"),
        ]:
            try:
                with winreg.OpenKey(hive, key) as k:
                    val, _ = winreg.QueryValueEx(k, "Path")
                    parts.append(val)
            except Exception:
                pass
        if parts:
            os.environ["PATH"] = os.pathsep.join(parts)
    except Exception:
        pass


def _run_in_thread_with_progress(fn, app, splash, start_prog, end_prog, label):
    """Run fn() in a thread. Poll every 33ms keeping the PyQt6 animation smooth."""
    done  = [False]
    error = [None]

    def _worker():
        try:
            fn()
        except Exception as e:
            error[0] = e
        finally:
            done[0] = True

    threading.Thread(target=_worker, daemon=True).start()

    prog = start_prog
    while not done[0]:
        prog = min(end_prog, prog + (end_prog - start_prog) * 0.015)
        splash.set_progress(prog, label)
        app.processEvents()
        time.sleep(0.033)

    if error[0]:
        raise error[0]


def _install_qt_excepthook():
    """Catch any exception that escapes during the event loop and surface it
    via the tkinter error dialog. Without this, pythonw silently eats Qt slot
    exceptions and the user just sees the splash disappear into nothing."""
    def _hook(exc_type, exc, tb):
        if issubclass(exc_type, KeyboardInterrupt):
            return
        msg = "".join(traceback.format_exception(exc_type, exc, tb))
        try:
            with open(os.path.join(HERE, "vueosd_crash.log"), "w") as f:
                f.write(msg)
        except Exception:
            pass
        _show_error("VueOSD crashed:\n\n" + msg[-4000:])
    sys.excepthook = _hook


def _run_with_splash():
    # ── Close the HTA — PyQt6 splash takes over immediately ───────────────────
    _hta_close()
    _install_qt_excepthook()

    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtGui     import QIcon

    sys.path.insert(0, HERE)
    from splash_screen import SplashScreen

    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setApplicationName("VueOSD")

    ico = os.path.join(HERE, "assets", "icon.png")
    if os.path.exists(ico):
        app.setWindowIcon(QIcon(ico))

    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                "VueOSD.OSDTool.1")
        except Exception:
            pass

    splash = SplashScreen()
    splash.show()

    def step(v, msg):
        splash.set_progress(v, msg)
        app.processEvents()

    step(0.04, "Checking dependencies…")

    if not _has("PIL"):
        step(0.08, "Installing Pillow…")
        _run_in_thread_with_progress(
            lambda: _pip("Pillow"), app, splash, 0.08, 0.18, "Installing Pillow…")

    step(0.18, "Checking NumPy…")
    if not _has("numpy"):
        _run_in_thread_with_progress(
            lambda: _pip("numpy"), app, splash, 0.18, 0.28, "Installing NumPy…")

    step(0.28, "Checking FFmpeg…")

    if sys.platform == "win32" and not _ffmpeg_ok():
        def _install_ffmpeg():
            try:
                subprocess.run(
                    ["winget", "install", "--id", "Gyan.FFmpeg",
                     "--source", "winget",
                     "--accept-package-agreements",
                     "--accept-source-agreements"],
                    capture_output=True, timeout=180)
            except Exception:
                pass
            _refresh_path_from_registry()

        _run_in_thread_with_progress(
            _install_ffmpeg, app, splash, 0.28, 0.58,
            "Installing FFmpeg (one-time, may take a minute)…")

        step(0.58, "FFmpeg ready" if _ffmpeg_ok() else "FFmpeg install failed")

    step(0.62, "Loading OSD parser…")
    step(0.70, "Loading font engine…")
    step(0.78, "Loading video pipeline…")
    step(0.86, "Building interface…")
    app.processEvents()

    import types
    mod = types.ModuleType("vueosd_main")
    mod.__file__ = os.path.join(HERE, "main.py")
    with open(mod.__file__, encoding="utf-8") as fh:
        src = fh.read()
    exec(compile(src, mod.__file__, "exec"), mod.__dict__)
    MainWindow = mod.MainWindow

    app.processEvents()
    step(0.94, "Almost ready…")
    app.processEvents()

    win = MainWindow()
    app.processEvents()

    splash.finish(win)
    sys.exit(app.exec())


def main():
    try:
        _hta_step(45, "Checking Python packages…")

        if not _has("PyQt6"):
            _hta_step(48, "Installing PyQt6 (one-time, ~80 MB)…")
            _pip("PyQt6")

        if not _has("PyQt6"):
            _hta_close()
            _show_error("Could not install PyQt6.\n\n"
                        "Check your internet connection and try again.")
            sys.exit(1)

        _run_with_splash()

    except Exception:
        _hta_close()
        _show_error("VueOSD failed to start:\n\n"
                    + traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
