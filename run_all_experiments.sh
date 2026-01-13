#!/bin/bash
################################################################################
# One-Click Experiment Runner for CIFAR-100 Optimization Experiments
#
# This script runs all 3 experiment sets with optimal GPU utilization
# Available GPUs: 0-7 (8 GPUs total)
################################################################################

set -e  # Exit on error

# Configuration
GPUS="0-7"
EPOCHS=100
SEED=42
SAVE_CHECKPOINTS=false
LOG_DIR="outputs/logs"
RESULTS_FILE="outputs/results/results.csv"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored messages
print_header() {
    echo -e "\n${BLUE}=================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}=================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Parse command line arguments
PARALLEL=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --parallel)
            PARALLEL=true
            shift
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --save-checkpoints)
            SAVE_CHECKPOINTS=true
            shift
            ;;
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --parallel           Run experiment sets in parallel (uses separate GPUs)"
            echo "  --epochs N           Number of epochs (default: 100)"
            echo "  --save-checkpoints   Save model checkpoints"
            echo "  --gpus RANGE         GPU IDs to use (default: 0-7)"
            echo "  --help               Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                           # Run all experiments sequentially"
            echo "  $0 --parallel                # Run experiments in parallel"
            echo "  $0 --epochs 200              # Run with 200 epochs"
            echo "  $0 --save-checkpoints        # Save model checkpoints"
            echo "  $0 --gpus 0-3                # Use only GPUs 0-3"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Build checkpoint flag
CHECKPOINT_FLAG=""
if [ "$SAVE_CHECKPOINTS" = true ]; then
    CHECKPOINT_FLAG="--save_checkpoints"
fi

# Print configuration
print_header "EXPERIMENT CONFIGURATION"
echo "GPU Configuration: ${GPUS}"
echo "Epochs: ${EPOCHS}"
echo "Seed: ${SEED}"
echo "Save Checkpoints: ${SAVE_CHECKPOINTS}"
echo "Parallel Execution: ${PARALLEL}"
echo "Results File: ${RESULTS_FILE}"

# Check if Python and required packages are available
print_header "CHECKING ENVIRONMENT"

if ! command -v python &> /dev/null; then
    print_error "Python not found. Please install Python."
    exit 1
fi
print_success "Python found"

if ! python -c "import torch" &> /dev/null; then
    print_error "PyTorch not found. Please install: pip install -r requirements.txt"
    exit 1
fi
print_success "PyTorch installed"

# Check GPU availability
NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")
if [ "$NUM_GPUS" -eq 0 ]; then
    print_warning "No GPUs detected. Will run on CPU (slow)."
else
    print_success "${NUM_GPUS} GPU(s) detected"
fi

# Backup existing results if they exist
if [ -f "$RESULTS_FILE" ]; then
    BACKUP_FILE="${RESULTS_FILE}.backup.$(date +%Y%m%d_%H%M%S)"
    print_warning "Backing up existing results to: ${BACKUP_FILE}"
    cp "$RESULTS_FILE" "$BACKUP_FILE"
fi

# Record start time
START_TIME=$(date +%s)
START_DATE=$(date '+%Y-%m-%d %H:%M:%S')

print_header "STARTING EXPERIMENTS"
echo "Start time: ${START_DATE}"

# Function to run a single experiment
run_experiment() {
    local EXP_NUM=$1
    local EXP_GPUS=$2
    local LOG_SUFFIX=$3

    print_header "EXPERIMENT SET ${EXP_NUM}"

    local CMD="python run_experiments.py --experiment ${EXP_NUM} --epochs ${EPOCHS} --gpus ${EXP_GPUS} ${CHECKPOINT_FLAG}"

    echo "Command: ${CMD}"

    if [ -n "$LOG_SUFFIX" ]; then
        $CMD > "outputs/logs/exp${EXP_NUM}_${LOG_SUFFIX}.txt" 2>&1 &
        local PID=$!
        echo "Running in background (PID: ${PID})"
        echo $PID
    else
        $CMD
        if [ $? -eq 0 ]; then
            print_success "Experiment ${EXP_NUM} completed successfully"
        else
            print_error "Experiment ${EXP_NUM} failed"
            return 1
        fi
    fi
}

# Run experiments based on mode
if [ "$PARALLEL" = true ]; then
    print_header "PARALLEL MODE: Running all experiments simultaneously"
    print_warning "This will use different GPU subsets for each experiment"

    # Split GPUs for parallel execution
    # Exp 1 (24 runs) -> GPUs 0-2 (3 GPUs)
    # Exp 2 (35 runs) -> GPUs 3-5 (3 GPUs)
    # Exp 3 (24 runs) -> GPUs 6-7 (2 GPUs)

    echo ""
    echo "GPU Allocation:"
    echo "  Experiment 1: GPUs 0-2 (24 runs)"
    echo "  Experiment 2: GPUs 3-5 (35 runs)"
    echo "  Experiment 3: GPUs 6-7 (24 runs)"
    echo ""

    PID1=$(run_experiment 1 "0-2" "parallel")
    PID2=$(run_experiment 2 "3-5" "parallel")
    PID3=$(run_experiment 3 "6-7" "parallel")

    print_success "All experiments launched in parallel"
    echo "PIDs: Exp1=${PID1}, Exp2=${PID2}, Exp3=${PID3}"
    echo ""
    echo "Monitoring progress... (Press Ctrl+C to cancel)"
    echo "Logs available in outputs/logs/"
    echo ""

    # Wait for all experiments to complete
    wait $PID1
    EXP1_STATUS=$?
    print_success "Experiment 1 completed"

    wait $PID2
    EXP2_STATUS=$?
    print_success "Experiment 2 completed"

    wait $PID3
    EXP3_STATUS=$?
    print_success "Experiment 3 completed"

    # Check if any failed
    if [ $EXP1_STATUS -ne 0 ] || [ $EXP2_STATUS -ne 0 ] || [ $EXP3_STATUS -ne 0 ]; then
        print_error "Some experiments failed. Check logs in outputs/logs/"
        exit 1
    fi

else
    print_header "SEQUENTIAL MODE: Running experiments one by one"
    print_warning "Each experiment will use all ${GPUS} GPUs"

    # Run experiments sequentially
    run_experiment 1 "$GPUS" "" || exit 1
    run_experiment 2 "$GPUS" "" || exit 1
    run_experiment 3 "$GPUS" "" || exit 1
fi

# Calculate elapsed time
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECONDS=$((ELAPSED % 60))

print_header "ALL EXPERIMENTS COMPLETED"
print_success "Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"

# Generate plots
print_header "GENERATING PLOTS"
python plot_results.py --stats

if [ $? -eq 0 ]; then
    print_success "Plots generated successfully"
    echo "  - outputs/plots/exp1_lr_ordering.png"
    echo "  - outputs/plots/exp2_eta_lambda_heatmap.png"
    echo "  - outputs/plots/exp3_batch_size_scaling.png"
else
    print_warning "Failed to generate plots. Run manually: python plot_results.py"
fi

# Print summary
print_header "SUMMARY"
echo "Results saved to: ${RESULTS_FILE}"
echo "Logs saved to: ${LOG_DIR}/"
if [ "$SAVE_CHECKPOINTS" = true ]; then
    echo "Checkpoints saved to: outputs/checkpoints/"
fi
echo "Plots saved to: outputs/plots/"

# Count results
if [ -f "$RESULTS_FILE" ]; then
    TOTAL_RUNS=$(tail -n +2 "$RESULTS_FILE" | wc -l)
    print_success "Total experiment runs: ${TOTAL_RUNS}"

    # Show best results for each experiment
    echo ""
    echo "Best Results:"
    python -c "
import pandas as pd
import sys

try:
    df = pd.read_csv('${RESULTS_FILE}')

    # Experiment 1
    exp1_methods = ['SGD', 'SGD+WD', 'SGDM+WD']
    for method in exp1_methods:
        df_method = df[df['method'] == method]
        if not df_method.empty:
            best_row = df_method.loc[df_method['best_test_acc'].idxmax()]
            print(f'  {method:10s}: {best_row[\"best_test_acc\"]:.2f}% (lr={best_row[\"lr\"]}, wd={best_row[\"wd\"]})')

    # Experiment 2
    df_sgdm = df[df['method'] == 'SGDM']
    if not df_sgdm.empty:
        best_row = df_sgdm.loc[df_sgdm['best_test_acc'].idxmax()]
        print(f'  SGDM Best : {best_row[\"best_test_acc\"]:.2f}% (lr={best_row[\"lr\"]}, wd={best_row[\"wd\"]})')

except Exception as e:
    print(f'  Could not analyze results: {e}', file=sys.stderr)
" 2>/dev/null || echo "  (Could not analyze results)"
fi

print_success "All done!"
echo ""
echo "To view plots:"
echo "  eog outputs/plots/*.png    # Linux"
echo "  open outputs/plots/*.png   # macOS"
echo ""
echo "To analyze results:"
echo "  python plot_results.py --stats"
echo ""
