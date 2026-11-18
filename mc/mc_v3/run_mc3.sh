#!/bin/bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
cd $SLURM_SUBMIT_DIR
python3 -c "import numpy" || { echo "NUMPY YOK, DURULDU"; exit 1; }
echo "Basladi: $(date) $(hostname)"
python3 run_mc3.py >> run_mc3.log 2>&1
echo "Bitti: $(date)"
[ -s mc3_raw.jsonl ] && python3 ozet3.py > ozet3.log 2>&1 && echo "TAMAM" || echo "mc3_raw.jsonl YOK - run_mc3.log'a bak"
