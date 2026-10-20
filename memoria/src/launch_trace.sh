#!/bin/bash
cd "$(dirname "$0")"; source /home/jhon/venv_fem/bin/activate 2>/dev/null
mkdir -p ../data/memory_figs/old_tol1e6; cp -n ../data/memory_figs/protocol_trace.npz ../data/memory_figs/old_tol1e6/ 2>/dev/null
F9_TOL=1e-9 F9_NC=2 python3 trace_memory_protocol.py > ../data/memory_figs/protocol_trace.log 2>&1
