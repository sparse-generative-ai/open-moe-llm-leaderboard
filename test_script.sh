OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python backend-cli.py  --debug \
                                                --task arena_hard \
                                                --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
                                                --precision bfloat16 \
                                                --inference-framework vllm_moe \
                                                --gpu-type NVIDIA-A100-SXM4-80GB \
                                                --batch_size 8

OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python backend-cli.py  --debug \
                                                --task arena_hard \
                                                --model mistralai/Mixtral-8x22B-Instruct-v0.1 \
                                                --precision bfloat16 \
                                                --inference-framework vllm_moe \
                                                --gpu-type NVIDIA-A100-SXM4-80GB \
                                                --batch_size 8

OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python backend-cli.py  --debug \
                                                --task arena_hard \
                                                --model databricks/dbrx-instruct \
                                                --precision bfloat16 \
                                                --inference-framework vllm_moe \
                                                --gpu-type NVIDIA-A100-SXM4-80GB \
                                                --batch_size 8

OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python backend-cli.py  --debug \
                                                --task arena_hard \
                                                --model Qwen/Qwen1.5-MoE-A2.7B-Chat \
                                                --precision bfloat16 \
                                                --inference-framework vllm_moe \
                                                --gpu-type NVIDIA-A100-SXM4-80GB \
                                                --batch_size 8