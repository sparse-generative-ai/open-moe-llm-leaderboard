#!/bin/bash
interval=10

# converter="/root/TensorRT-LLM/examples/dbrx/convert_checkpoint.py"
# checkpoint="databricks/dbrx-instruct"
# checkpoint="/root/.cache/huggingface/hub/models--databricks--dbrx-instruct/snapshots/c0a9245908c187da8f43a81e538e67ff360904ea"
# model_name="dbrx-instruct"

converter="/root/TensorRT-LLM/examples/llama/convert_checkpoint.py"
checkpoint="/root/.cache/huggingface/hub/models--mistralai--Mixtral-8x22B-Instruct-v0.1/snapshots/95d063951382d47385fe7b36e202b68639e5c066"
model_name="mixtral-8x22b-instruct-v0.1"


seq_len=4096
output_len=3000
bs=8
dtype="bfloat16"

# GPU显存阈值（MB）
memory_threshold=2048

while true; do
    # 使用 nvidia-smi 查询每个 GPU 的显存使用情况
    # --query-gpu=memory.used 查询已使用的显存大小
    # --format=csv,noheader,nounits 以 CSV 格式输出无标题无单位的结果
    mapfile -t memory_usage < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)

    # 假设所有 GPU 的显存使用都小于 1GB
    all_below_threshold=true

    # 遍历每个 GPU 的显存使用情况
    for mem in "${memory_usage[@]}"; do
        # 检查显存是否小于阈值
        if (( mem >= memory_threshold )); then
            all_below_threshold=false
            break
        fi
    done

    # 如果所有 GPU 的显存使用都小于 1GB，跳出循环
    if $all_below_threshold; then
        break
    fi

    # 等待指定的轮询间隔，并输出现在的时间
    echo "Waiting for GPU memory to be freed up... Current Time: $(date +"%Y-%m-%d %H:%M:%S")"
    sleep $interval
done

echo "Current Time: $(date +"%Y-%m-%d %H:%M:%S")"
trtllm-build --workers 2 --gpt_attention_plugin $dtype --gather_all_token_logits --max_batch_size $bs --max_input_len $seq_len --max_output_len $output_len --use_custom_all_reduce disable --tp_size 4 --pp_size 1 --checkpoint_dir $model_name-ckpt-int8 --output_dir $model_name-engine-int8 --gemm_plugin $dtype --weight_only_precision int8 --gpt_attention_plugin $dtype
echo "Current Time: $(date +"%Y-%m-%d %H:%M:%S")"
trtllm-build --workers 2 --gpt_attention_plugin $dtype --gather_all_token_logits --max_batch_size $bs --max_input_len $seq_len --max_output_len $output_len --use_custom_all_reduce disable --tp_size 4 --pp_size 1 --checkpoint_dir $model_name-ckpt-int4 --output_dir $model_name-engine-int4 --gemm_plugin $dtype --weight_only_precision int4 --gpt_attention_plugin $dtype
echo "Current Time: $(date +"%Y-%m-%d %H:%M:%S")"