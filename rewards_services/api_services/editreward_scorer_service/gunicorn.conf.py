import os
import sys


bind = f"{os.getenv('EDITREWARD_HOST', '127.0.0.1')}:{os.getenv('EDITREWARD_PORT', '18088')}"
workers = int(os.getenv("EDITREWARD_WORKERS", os.getenv("EDITREWARD_NUM_DEVICES", "1")))
worker_class = "sync"
timeout = int(os.getenv("EDITREWARD_TIMEOUT", "600"))

_raw_devices = os.getenv("EDITREWARD_CUDA_DEVICES") or os.getenv("CUDA_VISIBLE_DEVICES") or ""
CUDA_DEVICES = [device.strip() for device in _raw_devices.split(",") if device.strip()]
USED_DEVICES = set()


def pre_fork(server, worker):
    if not CUDA_DEVICES:
        return
    available = [device for device in CUDA_DEVICES if device not in USED_DEVICES]
    worker.cuda_device = available[0] if available else CUDA_DEVICES[len(USED_DEVICES) % len(CUDA_DEVICES)]
    USED_DEVICES.add(worker.cuda_device)
    print(f"Worker {worker.pid} assigned CUDA_VISIBLE_DEVICES={worker.cuda_device}", file=sys.stderr)


def post_fork(server, worker):
    cuda_device = getattr(worker, "cuda_device", None)
    if cuda_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device


def child_exit(server, worker):
    cuda_device = getattr(worker, "cuda_device", None)
    if cuda_device is not None:
        USED_DEVICES.discard(cuda_device)

