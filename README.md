# FSSDM

This directory contains the secure-inference code and plaintext inference/evaluation code used in the paper's supplementary material. It is split into two parts:

- `cipher-inference/`: secure-inference experiment code based on SHARK/C++.
- `plain-inference/`: plaintext inference, approximation baselines, ablation scripts, and metric scripts.

## Directory Structure

```text
code/
├── cipher-inference/
│   ├── DDPM1/
│   ├── DDPM2/
│   ├── SDUnCLIP1/
│   └── SDUnCLIP2/
└── plain-inference/
    ├── DDPM/
    └── SDUnCLIP/
```

## Setup

Each subdirectory under `cipher-inference` is an independent secure-inference project that can be built and run separately. Build from each project root:

```bash
cmake -DCMAKE_BUILD_TYPE=Release -S . -B build/
cmake --build build/ --config Release --target all -j
```

### cipher-inference/DDPM

Run DDPM as follows. Run DEALER first to generate offline preprocessing materials, then run SERVER and CLIENT separately.

```bash
  taskset -c 0-31 env SHARK_KEYBUF_IO_MB=512 OMP_NUM_THREADS=32 OMP_DYNAMIC=FALSE OMP_PROC_BIND=spread OMP_PLACES=cores GOMP_SPINCOUNT=0 OMP_WAIT_POLICY=PASSIVE GOMP_CPU_AFFINITY="0-31" ./build/benchmark-ddpm 2 --seed 20260304 --steps 5 --config  benchmarks/ddpm_bench_config.json

  taskset -c 0-15 env SHARK_KEYBUF_IO_MB=512 OMP_NUM_THREADS=16 OMP_DYNAMIC=FALSE OMP_PROC_BIND=close OMP_PLACES=cores GOMP_SPINCOUNT=0 OMP_WAIT_POLICY=PASSIVE GOMP_CPU_AFFINITY="0-15" ./build/benchmark-ddpm 0 127.0.0.1 42069 --seed 20260304 --steps 5  --config benchmarks/ddpm_bench_config.json > logs/server_online4.log 2>&1

  taskset -c 16-31 env SHARK_KEYBUF_IO_MB=512 OMP_NUM_THREADS=16 OMP_DYNAMIC=FALSE OMP_PROC_BIND=close OMP_PLACES=cores GOMP_SPINCOUNT=0 OMP_WAIT_POLICY=PASSIVE GOMP_CPU_AFFINITY="16-31" ./build/benchmark-ddpm 1 127.0.0.1 42069 --seed 20260304 --steps 5 --config benchmarks/ddpm_bench_config.json > logs/client_online4.log 2>&1
```


### cipher-inference/SDUnCLIP

Run SDUnCLIP similarly. Run DEALER first, then SERVER and CLIENT.

```bash
taskset -c 0-31 env SHARK_KEYBUF_IO_MB=512 OMP_NUM_THREADS=32 OMP_DYNAMIC=FALSE OMP_PROC_BIND=spread OMP_PLACES=cores OMP_WAIT_POLICY=PASSIVE GOMP_CPU_AFFINITY="0-31" UNCLIP_SEED=20260304 UNCLIP_ESTIMATE_REPEAT=1 ./build/benchmark-unclip_img2imgsmall 2 > logs/dealer_offline.log 2>&1

taskset -c 0-15 env SHARK_KEYBUF_IO_MB=512 OMP_NUM_THREADS=16 OMP_DYNAMIC=FALSE OMP_PROC_BIND=close OMP_PLACES=cores GOMP_SPINCOUNT=0 OMP_WAIT_POLICY=PASSIVE GOMP_CPU_AFFINITY="0-15" UNCLIP_SEED=20260304 UNCLIP_ESTIMATE_REPEAT=1 ./build/benchmark-unclip_img2imgsmall 0 127.0.0.1 42069 > logs/server_online1.log 2>&1

taskset -c 16-31 env SHARK_KEYBUF_IO_MB=512 OMP_NUM_THREADS=16 OMP_DYNAMIC=FALSE OMP_PROC_BIND=close OMP_PLACES=cores GOMP_SPINCOUNT=0 OMP_WAIT_POLICY=PASSIVE GOMP_CPU_AFFINITY="16-31" UNCLIP_SEED=20260304 UNCLIP_ESTIMATE_REPEAT=1 ./build/benchmark-unclip_img2imgsmall 1 127.0.0.1 42069 > logs/client_online1.log 2>&1
```


### plain-inference/DDPM

`plain-inference/DDPM/` includes training, plaintext inference, approximation inference, ablation scripts, and metric scripts for DDPM.

- `flax_train_tiny_mnist.py`: training script.
- `plain.py`: baseline inference.
- `plain_cipherdm.py`: CipherDM inference.
- `plain_fssdm.py`: FSSDM inference.
- `plain_ablation_jax.py`: ablation script.
- Other scripts: metric evaluation such as KID and SSIM.

### plain-inference/SDUnCLIP

`plain-inference/SDUnCLIP/` includes plaintext generation, approximation inference, and evaluation scripts for SDUnCLIP.

- `modify1/`: CipherDM approximation variant used in conda-based modification/runs.
- `modify2/`: FSSDM approximation variant used in conda-based modification/runs.
- `modify_1000.sh`: script for generation, inference, and evaluation pipeline.
- `run_all_unclip_once_fp32_no_xformers_no_safety_1000_det.py`: unified runner.
- `generate_cc3m_unclip_fast_fp32_no_xformers_no_safety_debug.py`: CC3M generation script.
- `generate_coco_unclip_fast_fp32_no_xformers_no_safety_debug.py`: COCO generation script.
- `generate_flickr30k_unclip_fast_fp32_no_xformers_no_safety_debug.py`: Flickr30k generation script.
- `eval_cc3m_unclip_fast_det.py`: CC3M evaluation script.
- `eval_coco_unclip_fast_det.py`: COCO evaluation script.
- `eval_flickr30k_unclip_fast_det.py`: Flickr30k evaluation script.
- `mse.py`: MSE metric script.

### Data&Model
The data and model are available at: https://drive.google.com/drive/folders/1xVpa_S-R7g7J1i9poQfyzOsrKMGx3Wdi?usp=sharing


### Used Open Source Projects 
This repository is built upon the following open-source projects:

```bash
https://github.com/kanav99/shark
https://github.com/Zhaoxinxinzi/CipherDM
https://github.com/huggingface/transformers
https://github.com/huggingface/transformers
```

