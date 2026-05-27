import os
# Workaround for PyOpenGL 3.1.6+ on Wayland: GLFW uses X11/XWayland but PyOpenGL
# defaults to Wayland EGL, causing a context mismatch. Force the X11 backend.
# Must be set before any `import OpenGL` happens.
if os.getenv("XDG_SESSION_TYPE") == "wayland" and not os.getenv("PYOPENGL_PLATFORM"):
    os.environ["PYOPENGL_PLATFORM"] = "x11"

import argparse
import sys

from loguru import logger


def _configure_logging(level: str, log_file: str | None):
    """Configure loguru sinks before any module imports it."""
    # Remove default stderr sink
    logger.remove()
    # Add stderr with chosen level
    logger.add(sys.stderr, level=level.upper(), colorize=True)
    # Optional file sink
    if log_file:
        logger.add(log_file, level=level.upper(), rotation="10 MB", retention=3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--colmap", type=str, default=None)
    parser.add_argument("--ply", type=str, default=None)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"],
                        help="Log level for stderr (and file if --log-file is set). Default: INFO")
    parser.add_argument("--log-file", type=str, default=None,
                        help="Optional file path to also write logs to")
    args = parser.parse_args()

    _configure_logging(args.log_level, args.log_file)

    from viewer.app import App
    app = App(width=args.width, height=args.height, colmap_path=args.colmap, ply_path=args.ply)
    app.run()


if __name__ == "__main__":
    main()
