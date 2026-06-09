"""
Entry point for the bundled executable (PyInstaller).
Handles multiprocessing freeze_support required by PyInstaller + Gradio.
"""
import os
import sys
import multiprocessing

# ── Fix PyInstaller paths ────────────────────────────────────────────────
if getattr(sys, 'frozen', False):
    # Running as bundled EXE — adjust paths
    BASE_DIR = os.path.dirname(sys.executable)
    # Ensure models package is discoverable
    models_dir = os.path.join(BASE_DIR, 'models')
    if os.path.isdir(models_dir) and models_dir not in sys.path:
        sys.path.insert(0, BASE_DIR)
    # Set checkpoint dir to be alongside the exe
    os.environ['APP_CHECKPOINT_DIR'] = os.path.join(BASE_DIR, 'checkpoints')
else:
    # Running as normal script
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

os.chdir(BASE_DIR)


def main():
    from app import build_interface

    # Show startup info
    print("=" * 60)
    print("  🏥 CancerGrading Assistant")
    print("=" * 60)
    print(f"  Working directory: {BASE_DIR}")

    # Check for checkpoints
    checkpoints_dir = os.environ.get('APP_CHECKPOINT_DIR',
                                     os.path.join(BASE_DIR, 'checkpoints'))
    if os.path.isdir(checkpoints_dir):
        pth_files = [f for f in os.listdir(checkpoints_dir) if f.endswith('.pth')]
        print(f"  Checkpoints found: {len(pth_files)}")
        for f in pth_files:
            print(f"    • {f}")
    else:
        print(f"  ⚠ No checkpoints directory found at: {checkpoints_dir}")
        print("  Please place .pth checkpoint files in a 'checkpoints' folder")
        print("  alongside the application executable.")

    print("=" * 60)
    print("  Starting server...")
    print("  Open http://localhost:7860 in your browser")
    print("=" * 60)

    demo = build_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
    )


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()