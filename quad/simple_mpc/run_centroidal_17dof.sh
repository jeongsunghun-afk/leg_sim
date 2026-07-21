#!/bin/bash
# 17-DOF B(centroidal) 안정 보행 — §5.5를 17-DOF 재현(2026-07-21). 기본값 박제됨(env 불요).
# 핵심: STIFF(접촉정합) + WAIST_LOCK(허리홀드) + REAR_LOCK(발목) + FOOT_DECISION(발자유) + WBVY(측방)
export CONDA_PREFIX=/home/jsh/simple-mpc/.pixi/envs/default
PIX=$CONDA_PREFIX/bin/python
cd "$(dirname "$0")"
VIEW=${VIEW:-0} VX=${VX:-0.2} STEPS=${STEPS:-1500} MAXITER=${MAXITER:-2} $PIX quad_centroidal_17dof.py "$@"
