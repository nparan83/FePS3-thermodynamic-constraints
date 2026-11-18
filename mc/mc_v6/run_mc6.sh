#!/bin/bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
cd $SLURM_SUBMIT_DIR
python3 -c "import numpy" || { echo "NUMPY YOK, DURULDU"; exit 1; }
echo "Basladi: $(date) $(hostname)"
python3 run_mc6.py >> run_mc6.log 2>&1
echo "Bitti: $(date)"
[ -s mc6_raw.jsonl ] && python3 ozet6.py > ozet6.log 2>&1 && echo "TAMAM" || echo "mc6_raw.jsonl YOK - run_mc6.log'a bak"
