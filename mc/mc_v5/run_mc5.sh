#!/bin/bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
cd $SLURM_SUBMIT_DIR
python3 -c "import numpy" || { echo "NUMPY YOK, DURULDU"; exit 1; }
echo "Basladi: $(date) $(hostname)"
python3 run_mc5.py >> run_mc5.log 2>&1
echo "Bitti: $(date)"
[ -s mc5_raw.jsonl ] && python3 ozet5.py > ozet5.log 2>&1 && echo "TAMAM" || echo "mc5_raw.jsonl YOK - run_mc5.log'a bak"
