model_name_or_path=/asr_lu/user/liuxiaohao/llms/THUDM/chatglm3-6b
checkpoint_dir=/asr_lu/user/liuxiaohao/project/LLM/LLaMA-Factory/output/xiaowei/chatglm3-6b-Qlora-trian/checkpoint-1008

python src/web_demo.py \
    --model_name_or_path ${model_name_or_path} \
    --template chatglm3 \
    --finetuning_type lora \
    --checkpoint_dir ${checkpoint_dir} \