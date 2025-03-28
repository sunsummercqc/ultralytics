import torch
torch.cuda.empty_cache()  # 清空CUDA缓存
print(f"清理后显存: {torch.cuda.memory_allocated()/1e9:.2f}GB")  # 验证清理效果