# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python record_experts.py --model_name mixtral22 --batch_size 8 --task arena
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python record_experts.py --model_name mixtral --batch_size 32 --task gsm8k
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python record_experts.py --model_name mixtral22 --batch_size 20 --task gsm8k
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python record_experts.py --model_name dbrx --batch_size 8 --task gsm8k
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python record_experts.py --model_name mixtral --batch_size 20 --task arena
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python record_experts.py --model_name mixtral22 --batch_size 5 --task arena
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python record_experts.py --model_name dbrx --batch_size 5 --task arena