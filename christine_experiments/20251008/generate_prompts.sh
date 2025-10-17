SCENARIOS_DIR=/workspace/charsci/christine_experiments/20251007/scenarios
PROMPTS_DIR=/workspace/charsci/christine_experiments/20251007/prompts
SCENARIOS_FILES=$(find $SCENARIOS_DIR -name "*_scenarios.jsonl")

for SCENARIOS_FILE in $SCENARIOS_FILES; do
    # Extract just the filename without path
    BASENAME=$(basename $SCENARIOS_FILE .jsonl)
    OUTPUT_FILE=$PROMPTS_DIR/${BASENAME}_prompts.jsonl

    echo "Generating prompts for $SCENARIOS_FILE -> $OUTPUT_FILE"
    python /workspace/charsci/src/model_written_evals/scenario/scenario_to_prompt.py \
        --input-file $SCENARIOS_FILE \
        --output-file $OUTPUT_FILE \
        --max-concurrent 40
done