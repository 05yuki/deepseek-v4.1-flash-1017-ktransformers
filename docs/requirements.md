# What it needs

RAM and VRAM play different parts here. RAM holds every routed expert and has to be
large enough: nothing in this setup makes the experts smaller, and with too little RAM
the server does not start. VRAM is different. Only the dense layers and the part of the
KV pool the attention scans in full have to fit on the GPUs: the KV pool always covers
the whole context, and the compressed KV that does not fit the GPU lives in host memory
([below](kv-cache.md#the-kv-pool-past-the-gpu)). The rest of the GPU memory is working room, measured
at launch, and each use of it shrinks when there is less:

- the prefill chunk is calibrated on the first launch to the memory that is left, so a
  smaller card runs smaller chunks. Every chunk streams the experts to the GPU again,
  so the prefill takes longer;
- during a long prefill, the memory of the GPU-resident weights and the KV pool's free
  pages is lent to it, and given back before decode;
- in decode, extra hot experts go into the KV pool's free pages and the driver's idle
  memory, as many as fit (`KT_KV_HOT_SPARE_FLOOR_MB` keeps a margin free); with less
  room fewer are kept, and decode is slower.

The KV pool itself is never made smaller to gain speed: the lends take only its free
pages on the GPU. If the server runs out of memory at launch, a shorter context
(`SGLANG_CONTEXT_LENGTH`) makes the pool smaller; it is the only setting memory needs.

- **RAM: a 512 GB class machine, or one socket with 256 GB.** The server uses about 420 GiB,
  most of it the MXFP4 CPU experts on 2 MB pages. The prefix cache's host copy adds about 5 GiB a GPU.
  - With one CPU expert pool on a NUMA node of less than 370 GiB, `serve.sh` keeps the
    experts in a RAM cache of the node's memory less 50 GiB and reads the others from the
    checkpoint when they are routed ([One socket, 256 GB](results.md#one-socket-256-gb)).
  - On two sockets, `KT_THREADPOOL_COUNT=1` with `numactl -N 0 -m 0` runs the server on
    one of them that way. The cache needs one pool: 384 GB on two sockets does not fit.

  The launcher makes one expert pool per NUMA node and uses every physical core.
  kt-kernel is built for the CPU it is built on (`CPUINFER_CPU_INSTRUCT=NATIVE`). Its
  MXFP4 kernels need AVX2.
- **GPU:** SM90 and newer on their own attention; SM89 (Ada, RTX 40) on a Triton
  fallback that keeps the V4 cache layout ([sm89.md](sm89.md)).
- **VRAM: 16 GB a card.** The dense layers and the KV pool's GPU part have to fit; the
  prefill chunks and the extra hot experts take the rest, so the cards show nearly full
  whatever room is left. `CUDA_VISIBLE_DEVICES` picks the GPUs, one tensor-parallel rank each.
  - Two 16 GB cards: 6.9 GB of weights each (half of the dense layers and one resident
    expert a layer) and the 1M-token KV pool.
  - One 16 GB card: 11.7 GB of weights, and the 1M-token KV pool too: its compressed KV
    (0.70 GiB at 1M tokens) goes to host memory.
  - Smaller cards do not hold their share of this checkpoint's dense layers with room for a
    long prefill: a more quantized checkpoint is needed (not covered here).
- **Disk: about 500 GB, NVMe.**
  - The checkpoint is 476 GiB (510 GB).
  - `.venv` is 8.6 GB and the upstream clones about 1 GB.
  - The load runs at the drive's speed, and the engram rows are read from it during
    inference, so a 1 TB NVMe drive is the practical choice.
- **Software.** Linux, bash 4.4 or newer, the CUDA toolkit (built and run with 13.1), git,
  gcc/g++, the libnuma and hwloc headers (`libnuma-dev libhwloc-dev`), and `uv` or Python 3.12.
  A system toolkit is used when it is CUDA 13.x (the pinned torch is built with CUDA 13).
  With glibc 2.42 or later it must also be 13.2 or newer, or a 13.1 updated since: older
  headers declare `rsqrt` differently from glibc and every DeepGEMM kernel build fails.
  Otherwise setup.sh downloads NVIDIA's toolkit wheels (CUDA 13.2, pinned in
  `lib/cuda-wheels.txt`, 1.3 GB) into `build/` and the kernels compiled at run time use those;
  nothing outside the checkout changes, and `CUDA_WHEELS=0` forbids the download. A prebuilt
  archive (`PREBUILT=`, below) takes the place of the builds, so then only gcc/g++, `libnuma1`
  and `libgomp1` are needed.
  The Python packages are pinned in `requirements.txt` (torch 2.14.1 + cu130): the tested
  venv's `pip freeze`, kept whole so that a build repeats it, measurement tools included.
- **Host settings.** Set these as root, persistently:
  - `echo advise > /sys/kernel/mm/transparent_hugepage/shmem_enabled`. The expert
    arenas want 2 MB pages; on 4 KB pages, pinning them takes most of an hour.
  - `vm.min_free_kbytes = 8388608`. Without it, the CUDA JIT can hit a node-local OOM
    while the experts fill the nodes.
  - Swap off.
- **In a container.** The host settings above are set on the host. The recipe was tested in
  Docker on plain Ubuntu 24.04 and 26.04 images with the NVIDIA Container Toolkit, run as
  `docker run --gpus all --network host --shm-size 32g --ulimit memlock=-1 --cap-add SYS_NICE`.
  `--cap-add SYS_NICE` is required: Docker's default seccomp profile refuses NUMA memory-policy
  calls such as `set_mempolicy` without it. With a prebuilt archive, the image needs only git,
  curl, ca-certificates, gcc/g++, `libnuma1`, `libgomp1` and uv.
