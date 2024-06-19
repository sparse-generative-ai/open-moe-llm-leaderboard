CUDA_VISIBLE_DEVICES=0,1,2,3 python record_experts.py --model_name mixtral22 --batch_size 8 --task arena
CUDA_VISIBLE_DEVICES=0,1,2,3 python record_experts.py --model_name dbrx --batch_size 8 --task arena
CUDA_VISIBLE_DEVICES=0 python record_experts.py --model_name qwen --batch_size 16 --task arena