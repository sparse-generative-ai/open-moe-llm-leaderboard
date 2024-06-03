converter="/root/TensorRT-LLM/examples/llama/convert_checkpoint.py"
# checkpoint="mistralai/Mixtral-8x7B-instruct-v0.1"
checkpoint="/root/.cache/huggingface/hub/models--mistralai--mixtral-8x7b-instruct-v0.1/snapshots/1e637f2d7cb0a9d6fb1922f305cb784995190a83"
model_name="mixtral-8x7b-instruct-v0.1"

seq_len=4096
bs=16
dtype="bfloat16"

# python /root/TensorRT-LLM/examples/llama/convert_checkpoint.py --workers 4 --tp_size 4 --model_dir mistralai/Mixtral-8x7B-instruct-v0.1 --output_dir mixtral-8x7b-v0.1-instruct-ckpt-bfloat16 --dtype bfloat16

# python $converter --workers 4 --tp_size 4 --model_dir $checkpoint --output_dir $model_name-ckpt-bfloat16 --dtype bfloat16 
# python $converter --tp_size 4 --model_dir $checkpoint --output_dir $model_name-ckpt-int8 --dtype bfloat16 --use_weight_only --weight_only_precision int8
# python $converter --tp_size 4 --model_dir $checkpoint --output_dir $model_name-ckpt-int4 --dtype bfloat16 --use_weight_only --weight_only_precision int4

# trtllm-build --workers 4 --gpt_attention_plugin bfloat16 --gather_all_token_logits --max_batch_size $bs --max_input_len $seq_len --use_custom_all_reduce disable --tp_size 4 --pp_size 1 --checkpoint_dir $model_name-ckpt-bfloat16 --output_dir $model_name-engine-bfloat16 --gemm_plugin bfloat16 --gpt_attention_plugin bfloat16
trtllm-build --workers 2 --gpt_attention_plugin bfloat16 --gather_all_token_logits --max_batch_size $bs --max_input_len $seq_len --use_custom_all_reduce disable --tp_size 4 --pp_size 1 --checkpoint_dir $model_name-ckpt-int8 --output_dir $model_name-engine-int8 --gemm_plugin bfloat16 --weight_only_precision int8 --gpt_attention_plugin bfloat16
trtllm-build --workers 2 --gpt_attention_plugin bfloat16 --gather_all_token_logits --max_batch_size $bs --max_input_len $seq_len --use_custom_all_reduce disable --tp_size 4 --pp_size 1 --checkpoint_dir $model_name-ckpt-int4 --output_dir $model_name-engine-int4 --gemm_plugin bfloat16 --weight_only_precision int4 --gpt_attention_plugin bfloat16

# converter="/root/TensorRT-LLM/examples/dbrx/convert_checkpoint.py"
# checkpoint="databricks/dbrx-instruct"
# checkpoint="/root/.cache/huggingface/hub/models--databricks--dbrx-instruct/snapshots/c0a9245908c187da8f43a81e538e67ff360904ea"
# model_name="dbrx-instruct"

# python $converter --workers 4 --tp_size 4 --model_dir $checkpoint --output_dir $model_name-ckpt-bfloat16 --dtype $dtype
# python $converter --workers 4 --tp_size 4 --model_dir $checkpoint --output_dir $model_name-ckpt-int8 --dtype $dtype --use_weight_only --weight_only_precision int8
# python $converter --workers 4 --tp_size 4 --model_dir $checkpoint --output_dir $model_name-ckpt-int4 --dtype $dtype --use_weight_only --weight_only_precision int4

# trtllm-build --workers 4 --gpt_attention_plugin $dtype --gather_all_token_logits --max_batch_size $bs --max_input_len $seq_len --use_custom_all_reduce disable --tp_size 4 --pp_size 1 --checkpoint_dir $model_name-ckpt-bfloat16 --output_dir $model_name-engine-bfloat16 --gemm_plugin $dtype
# trtllm-build --workers 4 --gpt_attention_plugin $dtype --gather_all_token_logits --max_batch_size $bs --max_input_len $seq_len --use_custom_all_reduce disable --tp_size 4 --pp_size 1 --checkpoint_dir $model_name-ckpt-int8 --output_dir $model_name-engine-int8 --gemm_plugin $dtype --weight_only_precision int8
# trtllm-build --workers 4 --gpt_attention_plugin $dtype --gather_all_token_logits --max_batch_size $bs --max_input_len $seq_len --use_custom_all_reduce disable --tp_size 4 --pp_size 1 --checkpoint_dir $model_name-ckpt-int4 --output_dir $model_name-engine-int4 --gemm_plugin $dtype --weight_only_precision int4
