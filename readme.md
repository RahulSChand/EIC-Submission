Code to run the experiments

# Dependencies

pip install transformers
pip install datasets

# Note

The pre-trained GPT2 compatible model weights to run the below can be downloaded from:
https://drive.google.com/file/d/1VV-R6-bxdSplL-ySR5HtRs--xYOzpXBd/view?usp=sharing

Download & save it as "pytorch_model.bin"

These weights are same as the ones from huggingface except the weights are transposed for ease of implementation of Quant & use on nn.Lienar (since default hugginface gpt2 has nn.Conv2D in the implementation)

# Run only lora + LLM-QAT (The LLM-QAT code inside utils_qat.py taken from https://github.com/facebookresearch/LLM-QAT/blob/main/models/utils_quant.py)

./train_qa.sh

# --- The command that runs in train_qa.sh is below ---
--is_lora = if use lora
--a_bit = activation quantization bit width
--w_bit = weight quantization bit width
--only-eval = used to evaluate
--eval_model_name = model name if you want to evaluate

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
    --eval_model_name model.bin \
    --disable_wandb 0

# --------------------

# For switch precision training

./train_switch.sh

To change the bit-widths used in switch change line 869/870 in run_qa_lora_quant_switch.py

# For cycle precision training

./train_cycle.sh


# Other files

The other files are 
custom_gpt2_lora_switch.py -> Model implementation with LoRA for each linear layer, LLM-QAT quantization & different quantziation for different layers




