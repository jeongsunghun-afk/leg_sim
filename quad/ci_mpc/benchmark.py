"""Phase 1 benchmark: quantify the fixed-schedule OCP baseline (ci_mpc).

Sweeps forward speed and tabulates falls / distance / mean vx / joint torque peaks
/ solve time. Each speed runs ci_mpc_run.py as a fresh subprocess (clean state).

This is the model-based OCP baseline for the eventual A(WBIC) vs OCP vs CI-MPC
comparison. Reference joint-load numbers for A are in memory [[joint-load-trot-walk]].

Usage: MJPY benchmark.py   [SPEEDS="0.2 0.3 0.4 0.5"]  [STEPS=2500]
"""
import os, re, subprocess, sys

PY = sys.executable
HERE = os.path.dirname(os.path.abspath(__file__))
SPEEDS = [float(s) for s in os.environ.get("SPEEDS", "0.2 0.3 0.4 0.5 0.6").split()]
STEPS = os.environ.get("STEPS", "2500")
ENV = dict(os.environ, KP="150", KD="5", FF="1", MAXIT="20", STEPS=STEPS, VIEW="0")

FIELDS = ["falls", "x", "z", "vx_mean", "tau_thigh", "tau_calf", "tau_foot", "sat_pct", "solve_ms"]
print(f"{'VX':>5} {'falls':>5} {'dist':>7} {'z':>6} {'vx':>7} "
      f"{'τthigh':>7} {'τcalf':>6} {'τfoot':>6} {'sat%':>5} {'solve_ms':>8}")
print("-" * 72)
for vx in SPEEDS:
    out = subprocess.run([PY, "ci_mpc_run.py"], cwd=HERE,
                         env=dict(ENV, VX=str(vx)), capture_output=True, text=True)
    line = out.stdout.strip().splitlines()[-1] if out.stdout.strip() else ""
    m = {k: (re.search(rf"{k}=([-+0-9.]+)", line).group(1)
             if re.search(rf"{k}=([-+0-9.]+)", line) else "?") for k in FIELDS}
    print(f"{vx:>5} {m['falls']:>5} {m['x']:>7} {m['z']:>6} {m['vx_mean']:>7} "
          f"{m['tau_thigh']:>7} {m['tau_calf']:>6} {m['tau_foot']:>6} {m['sat_pct']:>5} {m['solve_ms']:>8}")
