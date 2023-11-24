model_name_or_path=/asr_lu/llms/GLM/chatglm2-6b
epoch=50
output_dir=/asr_lu/user/liuxiaohao/project/LLM/LLaMA-Efficient-Tuning/output/xiaowei_cmp/chatglm2-6b-Qlora-trian

deepspeed_config_file=/asr_lu/user/liuxiaohao/project/LLM/LLaMA-Efficient-Tuning/scrips/ds_zero2_no_offload.json
CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage sft \
    --model_name_or_path ${model_name_or_path} \
    --do_train \
    --dataset XiaoweiCMP \
    --template chatglm2 \
    --finetuning_type lora \
    --output_dir ${output_dir} \
    --overwrite_cache \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --lr_scheduler_type cosine \
    --logging_steps 16 \
    --save_steps 16 \
    --learning_rate 5e-5 \
    --num_train_epochs ${epoch} \
    --plot_loss \
    --fp16 \
    --lora_target query_key_value \
    --max_source_length 1024 \
    --max_target_length 1024 \
    --quantization_bit 4