CUDA_LAUNCH_BLOCKING=1 python run_qa_lora_quant_cycle.py \
    --dataset_name squad \
    --max_seq_length 384 \
    --model_name_or_path gpt2 \
    --per_device_train_batch_size 26 \
    --per_device_eval_batch_size 240 \
    --pad_to_max_length \
    --learning_rate 2e-4 \
    --p_name switch_nodistil_32_8_4_CYCLIC_v2 \
    --is_lora 1 \
    --half 0 \
    --only_eval 0 \
    --a_bit 4 \
    --w_bit 4 \
    --disable_wandb 0 \
    --num_train_epochs 6 \