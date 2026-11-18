#!/bin/bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
cd $SLURM_SUBMIT_DIR
python3 -c "import numpy" || { echo "NUMPY YOK, DURULDU"; exit 1; }
echo "Basladi: $(date) $(hostname)"
python3 run_mc7.py >> run_mc7.log 2>&1
echo "Bitti: $(date)"
[ -s mc7_raw.jsonl ] && python3 ozet7.py > ozet7.log 2>&1 && echo "TAMAM" || echo "mc7_raw.jsonl YOK - run_mc7.log'a bak"
