"""
Build a standalone Windows executable for CancerGrading Assistant.
Run:  python build_exe.py

Output: ./dist/CancerGrading Assistant.exe
"""
import os
import sys
import shutil
import site
import subprocess


def main():
    # Ensure we're in the apptest directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    print("=" * 60)
    print("  🏥 Building CancerGrading Assistant EXE")
    print("=" * 60)

    # ── Step 1: Verify dependencies ────────────────────────────────────
    print("\n[1/4] Checking dependencies...")
    try:
        import torch
        import gradio
        import PIL
        print(f"  ✓ torch {torch.__version__}")
        print(f"  ✓ gradio {gradio.__version__}")
        print(f"  ✓ PIL {PIL.__version__}")
    except ImportError as e:
        print(f"  ✗ Missing dependency: {e}")
        print("  Run: pip install -r requirements.txt")
        sys.exit(1)

    # ── Step 2: Ensure PyInstaller is installed ────────────────────────
    print("\n[2/4] Ensuring PyInstaller is available...")
    try:
        import PyInstaller
        print(f"  ✓ PyInstaller {PyInstaller.__version__}")
    except ImportError:
        print("  Installing PyInstaller...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
        print("  ✓ PyInstaller installed")

    # ── Step 3: Clean previous builds ──────────────────────────────────
    print("\n[3/4] Cleaning previous builds...")
    for folder in ['build', 'dist']:
        if os.path.isdir(folder):
            shutil.rmtree(folder)
            print(f"  ✓ Removed {folder}/")

    # ── Step 4: Run PyInstaller ────────────────────────────────────────
    print("\n[4/4] Building executable (this may take several minutes)...")
    print("  This will bundle Python, PyTorch, Gradio, and all models.\n")

    # Collect hidden imports needed by torch, gradio, etc.
    hidden_imports = [
        'torch',
        'torchvision',
        'gradio',
        'PIL',
        'PIL._imaging',
        'numpy',
        'models',
        'models.alexnet',
        'models.vgg16',
        'models.vgg19',
        'models.resnet50',
        'models.fusion',
        'models.attention',
    ]

    # Data files to bundle: checkpoints + models source
    datas = [
        ('checkpoints', 'checkpoints'),
        ('models/*.py', 'models'),
    ]

    # Build command
    cmd = [
        sys.executable, '-m', 'PyInstaller',
        '--name=CancerGrading Assistant',
        '--onefile',                      # Single EXE
        '--windowed',                     # No console window (optional)
        '--console',                      # Keep console for now (debug)
        '--clean',
        '--noconfirm',
        f'--distpath=dist',
        f'--workpath=build',
        '--add-data=checkpoints;checkpoints',
        '--add-data=models;models',
        # Collect all submodules
        '--collect-all=torch',
        '--collect-all=torchvision',
        '--collect-all=gradio',
        '--collect-all=PIL',
        '--collect-all=numpy',
        '--collect-all=scipy',
        # Hidden imports
    ]
    for mod in hidden_imports:
        cmd.append(f'--hidden-import={mod}')

    # Add entry point
    cmd.append('run_app.py')

    # Run PyInstaller
    subprocess.check_call(cmd)

    # ── Final ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  ✅ BUILD COMPLETE!")
    print("=" * 60)

    exe_name = "CancerGrading Assistant.exe"
    exe_path = os.path.join(script_dir, "dist", exe_name)
    if os.path.isfile(exe_path):
        size_mb = os.path.getsize(exe_path) / (1024 * 1024)
        print(f"\n  Executable created: {exe_path}")
        print(f"  Size: {size_mb:.1f} MB")
        print(f"\n  To distribute:")
        print(f"  1. Copy '{exe_name}' to any Windows computer")
        print(f"  2. Copy the 'checkpoints/' folder alongside it")
        print(f"  3. Run the EXE — no Python installation needed!")
    else:
        print(f"\n  ✗ Build may have failed — check for errors above.")
        print(f"  Expected: {exe_path}")


if __name__ == '__main__':
    main()