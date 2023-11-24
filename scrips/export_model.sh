model_name_or_path=/asr_lu/user/liuxiaohao/project/llms/GLM/chatglm2-6b
# checkpoint_dir=/asr_lu/user/liuxiaohao/project/LLM/LLaMA-Efficient-Tuning/output/1_epoch
# output_dir=/asr_lu/user/liuxiaohao/project/LLM/LLaMA-Efficient-Tuning/output/1_epoch/export_model
checkpoint_dir=/asr_lu/user/liuxiaohao/project/LLM/LLaMA-Efficient-Tuning/output/personality/chatglm2-6b-Qlora-trian/checkpoint-1050
output_dir=/asr_lu/user/liuxiaohao/project/LLM/LLaMA-Efficient-Tuning/output/personality/export-models/chatglm2-6b-Qlora-trian/checkpoint-1050-export
python src/export_model.py \
    --model_name_or_path ${model_name_or_path} \
    --template chatglm2-6b \
    --finetuning_type lora \
    --checkpoint_dir ${checkpoint_dir} \
    --output_dir ${output_dir} \
