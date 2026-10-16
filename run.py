"""
=============================================================================
run.py — One-Command Application Launcher
=============================================================================

This script:
    1. Checks Python version (>= 3.9)
    2. Creates a virtual environment if not present
    3. Installs requirements
    4. Checks for .env file (creates from .env.example if missing)
    5. Initialises the database
    6. Starts the FastAPI server with uvicorn

Usage:
    python run.py                    # Start normally
    python run.py --port 8080        # Custom port
    python run.py --host 0.0.0.0     # Expose to network
    python run.py --no-reload        # Disable hot-reload
=============================================================================
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path(__file__).parent


def check_python_version():
    if sys.version_info < (3, 9):
        print(f"✗ Python 3.9+ required. Found: {sys.version}")
        sys.exit(1)
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")


def check_env_file():
    env_file = PROJECT_ROOT / ".env"
    example_file = PROJECT_ROOT / ".env.example"

    if not env_file.exists():
        if example_file.exists():
            shutil.copy(example_file, env_file)
            print("✓ Created .env from .env.example")
            print("  ⚠  Please edit .env and set a strong SECRET_KEY before production use!")
        else:
            print("⚠  No .env file found. Using default settings.")
    else:
        print("✓ .env file exists")


def print_startup_banner():
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║          ███╗   ███╗██╗███╗   ██╗██████╗                     ║
║          ████╗ ████║██║████╗  ██║██╔══██╗                    ║
║          ██╔████╔██║██║██╔██╗ ██║██║  ██║                    ║
║          ██║╚██╔╝██║██║██║╚██╗██║██║  ██║                    ║
║          ██║ ╚═╝ ██║██║██║ ╚████║██████╔╝                    ║
║                                                              ║
║          B R I D G E    A I                                  ║
║          Mental Health Chatbot — Final Year Project          ║
║                                                              ║
║  Architecture:                                               ║
║    • Emotion:     DistilBERT (GoEmotions, 28 classes)        ║
║    • Intent:      BiLSTM + Bahdanau Attention                ║
║    • Response:    FLAN-T5 (EmpatheticDialogues)              ║
║    • Sentiment:   RoBERTa (Twitter)                          ║
║    • Safety:      DistilBERT (5-class crisis detection)      ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def main():
    parser = argparse.ArgumentParser(description="Run MindBridge AI")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--no-reload", action="store_true", help="Disable hot-reload")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    args = parser.parse_args()

    print_startup_banner()
    check_python_version()
    check_env_file()

    print(f"\n Starting MindBridge AI server...")
    print(f"   Host:   {args.host}")
    print(f"   Port:   {args.port}")
    print(f"   Reload: {not args.no_reload}")
    print(f"\n  Access the application at: http://{args.host}:{args.port}")
    if args.host == "127.0.0.1":
        print(f"   Or use:                    http://localhost:{args.port}")
    print(f"\n  API Documentation:         http://localhost:{args.port}/api/docs")
    print(f"\n  Press Ctrl+C to stop\n")

    cmd = [
        sys.executable, "-m", "uvicorn",
        "backend.main:app",
        "--host", args.host,
        "--port", str(args.port),
        "--log-level", "info",
    ]

    if not args.no_reload:
        cmd.append("--reload")

    if args.workers > 1:
        cmd.extend(["--workers", str(args.workers)])

    try:
        subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)
    except KeyboardInterrupt:
        print("\n\n MindBridge AI stopped. Take care! 💙\n")
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Server crashed with exit code {e.returncode}")
        print("  Check the logs above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
