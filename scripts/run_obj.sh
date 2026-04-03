#!/bin/bash

BASE_GPU=0
BASE_CONCEPT="golf_ball"


OUTPUT_DIR="./model-infinity/concept-$BASE_CONCEPT/log_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"
LOG_FILE="$OUTPUT_DIR/experiment_log.txt"
echo "Experiment started at $(date)" > "$LOG_FILE"


        
infer4code_params=(
    "--gpu" "$BASE_GPU"
    "--concept" "golf_ball"
    "--pr" "golf_ball"
    "--cfg" "2.5"
    "--multi" "3"
    "--attn" "35"
    "--ssi" "2"
    "--esi" "9"
)

OUTPUT=$(python infer4code.py "${infer4code_params[@]}" 2>&1)

echo "$OUTPUT" >> "$LOG_FILE"
echo "----------------------------------------" >> "$LOG_FILE"


CODE_PATH=$(echo "$OUTPUT" | grep "Code saved to:" | awk '{print $4}')
echo "$CODE_PATH"

            

infer4eval_para=(
    "--gpu" "$BASE_GPU"
    "--concept" "golf_ball"
    "--code_path" "$CODE_PATH"
    "--alpha" "3.5"
    "--pr" "golf_ball_eval"
    "--ssi" "2"
    "--esi" "9"
    "--replace_code" "0"
)

IMG_OUTPUT=$(python infer4eval.py "${infer4eval_para[@]}" 2>&1)
IMG_PATH=$(echo "$IMG_OUTPUT" | grep "Image saved to:" | awk '{print $4}')

                                    

echo "All experiments completed. Total runs:" | tee -a "$LOG_FILE"
echo "Summary of results saved to: $OUTPUT_DIR/results_summary.csv" | tee -a "$LOG_FILE"