"""Check that Numba can find and run kernels on the GPU."""

import numpy as np
from numba import cuda


@cuda.jit
def add_one(arr):
    i = cuda.grid(1)
    if i < arr.size:
        arr[i] += 1.0


def main():
    if not cuda.is_available():
        print("No usable CUDA GPU found. Check the NVIDIA driver and the numba-cuda install.")
        return
    cuda.detect()

    dev = cuda.get_current_device()
    free, total = cuda.current_context().get_memory_info()
    print(f"\nUsing: {dev.name} (compute capability {dev.compute_capability[0]}.{dev.compute_capability[1]})")
    print(f"GPU memory: {free / 2**30:.1f} GB free of {total / 2**30:.1f} GB")

    host = np.arange(1_000_000, dtype=np.float32)
    dev_arr = cuda.to_device(host)
    threads = 256
    blocks = (host.size + threads - 1) // threads
    add_one[blocks, threads](dev_arr)
    ok = np.array_equal(dev_arr.copy_to_host(), host + 1)
    print(f"Test kernel ({blocks} blocks x {threads} threads): {'PASSED' if ok else 'FAILED'}")


if __name__ == "__main__":
    main()
