import torch


# Check available devices
def check_gpu_device():
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"Number of GPUs available: {device_count}")

        for i in range(device_count):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memory Allocated: {torch.cuda.memory_allocated(i)} bytes")
            print(f"  Memory Cached: {torch.cuda.memory_reserved(i)} bytes")
    else:
        print("CUDA is not available on this system.")


# Run the check
check_gpu_device()
