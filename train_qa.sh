CUDA_LAUNCH_BLOCKING=1 python run_qa_quant_lora.py \
    --dataset_name squad \
    --max_seq_length 384 \
    --model_name_or_path gpt2 \
    --per_device_train_batch_size 26 \
    --per_device_eval_batch_size 32 \
    --pad_to_max_length \
    --learning_rate 2e-4 \
    --p_name bit_4_16_lora_rnd \
    --is_lora 1 \
    --half 0 \
    --only_eval 0 \
    --a_bit 16 \
    --w_bit 4 \
    --disable_wandb 0

