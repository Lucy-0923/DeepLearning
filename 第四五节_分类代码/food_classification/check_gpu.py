import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch

print("=== GPU状态检查 ===")
print("CUDA是否可用:", torch.cuda.is_available())
print("GPU数量:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("当前GPU索引:", torch.cuda.current_device())
    print("GPU名称:", torch.cuda.get_device_name(0))
    print("GPU内存:", torch.cuda.get_device_properties(0).total_memory / 1024**3, "GB")
else:
    print("警告：未检测到GPU，将使用CPU进行训练")
