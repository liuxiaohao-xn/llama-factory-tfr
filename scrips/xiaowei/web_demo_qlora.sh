# chatglm2
# model_name_or_path=/asr_lu/llms/GLM/chatglm2-6b
# checkpoint_dir=/asr_lu/user/liuxiaohao/project/LLM/LLaMA-Efficient-Tuning/output/xiaowei_cmp/chatglm2-6b-Qlora-trian/checkpoint-672
#template=chatglm2


# baichuan2
model_name_or_path=/asr_lu/llms/Baichuan/Baichuan2-13B-Chat
checkpoint_dir=/asr_lu/user/liuxiaohao/project/LLM/LLaMA-Efficient-Tuning/output/xiaowei_cmp/Baichuan2-13B-Chat-qlora-train/checkpoint-128
template=baichuan2

python src/web_demo.py \
    --model_name_or_path ${model_name_or_path} \
    --template ${template} \
    --finetuning_type lora \
    --checkpoint_dir ${checkpoint_dir} \
    --quantization_bit 4
