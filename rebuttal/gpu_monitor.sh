#!/bin/bash
# Monitor GPUs 2,3 and start VGG-16 seed=123 when they become free
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate

EPOCHS=100
WPG=8
CHECK_INTERVAL=60  # seconds

echo "[$(date)] GPU Monitor started. Watching GPUs 2,3..."

while true; do
    # Get memory usage for GPU 2 and 3
    MEM2=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 2 | tr -d ' ')
    MEM3=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 3 | tr -d ' ')

    # Consider "free" if memory < 1000 MiB
    FREE2=0
    FREE3=0
    [ "$MEM2" -lt 1000 ] 2>/dev/null && FREE2=1
    [ "$MEM3" -lt 1000 ] 2>/dev/null && FREE3=1

    if [ "$FREE2" -eq 1 ] && [ "$FREE3" -eq 1 ]; then
        echo "[$(date)] Both GPU 2,3 are free (mem: ${MEM2}MiB, ${MEM3}MiB). Starting VGG-16 seed=123 on GPUs 2,3..."
        
        for EXP in 1 2 3; do
            echo "[$(date)] VGG-16 seed=123 Exp$EXP START (GPUs 2,3)"
            python3 rebuttal/run_rebuttal.py --model vgg16 --experiment $EXP --seed 123 --gpus 2,3 --epochs $EPOCHS --workers_per_gpu $WPG
            echo "[$(date)] VGG-16 seed=123 Exp$EXP DONE"
        done
        
        echo "[$(date)] VGG-16 seed=123 ALL DONE"
        break
    elif [ "$FREE2" -eq 1 ]; then
        echo "[$(date)] Only GPU 2 free (mem: ${MEM2}MiB). GPU 3 still busy (${MEM3}MiB). Starting on GPU 2 alone..."
        
        for EXP in 1 2 3; do
            echo "[$(date)] VGG-16 seed=123 Exp$EXP START (GPU 2)"
            python3 rebuttal/run_rebuttal.py --model vgg16 --experiment $EXP --seed 123 --gpus 2 --epochs $EPOCHS --workers_per_gpu $WPG
            echo "[$(date)] VGG-16 seed=123 Exp$EXP DONE"
        done
        
        echo "[$(date)] VGG-16 seed=123 ALL DONE (GPU 2 only)"
        break
    elif [ "$FREE3" -eq 1 ]; then
        echo "[$(date)] Only GPU 3 free (mem: ${MEM3}MiB). GPU 2 still busy (${MEM2}MiB). Starting on GPU 3 alone..."
        
        for EXP in 1 2 3; do
            echo "[$(date)] VGG-16 seed=123 Exp$EXP START (GPU 3)"
            python3 rebuttal/run_rebuttal.py --model vgg16 --experiment $EXP --seed 123 --gpus 3 --epochs $EPOCHS --workers_per_gpu $WPG
            echo "[$(date)] VGG-16 seed=123 Exp$EXP DONE"
        done
        
        echo "[$(date)] VGG-16 seed=123 ALL DONE (GPU 3 only)"
        break
    else
        echo "[$(date)] GPUs 2,3 busy (mem: ${MEM2}MiB, ${MEM3}MiB). Checking again in ${CHECK_INTERVAL}s..."
    fi

    sleep $CHECK_INTERVAL
done
