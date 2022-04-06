GLUE_DIR=../data/glue_data
TASK_NAME=STS-B
OUTPUT_DIR=../out/glue/$TASK_NAME/
python src/glue/get_x0.py \
    --model_name_or_path albert-large-v2 \
    --task_name STS-B \
    --max_seq_length 64 \
    --do_lower_case \
    --data_dir $GLUE_DIR/$TASK_NAME \
    --overwrite_output_dir \
    --output_dir $OUTPUT_DIR \
    --init_model 0 \
    --num_layer 24 \
    --output_hidden_states