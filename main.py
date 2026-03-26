# main.py
# ─────────────────────────────────────────────────────────────────────────────
# Application entry point for the Autonomous Scientific Experiment Planner.
#
# This file is intentionally minimal — its only job is to import and call
# the launch function from the UI module. All application logic lives in
# the agents/, core/, tools/, and ui/ directories.
#
# Run with: python main.py
# Then open your browser to: http://localhost:7860
# ─────────────────────────────────────────────────────────────────────────────

from ui.app import launch_app

if __name__ == "__main__":
    launch_app()