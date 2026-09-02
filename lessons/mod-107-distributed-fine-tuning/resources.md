# Resources for mod-107-distributed-fine-tuning

Curated primary references. Prefer these — the paper of record, the official docs, the current tool README — over blog-post summaries when you go deep. Version pins move; check the current compatibility matrix before committing dependencies.

## Chapter 1 — The parallelism landscape

### Papers

- **Rajbhandari et al., 2020** — "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models." The paper of record for the ZeRO-1 / 2 / 3 memory accounting. Read section 3 (the memory math) and section 5 (the compute / communication analysis). https://arxiv.org/abs/1910.02054
- **Zhao et al., 2023** — "PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel." The PyTorch team's paper describing FSDP1's design, wrapping strategies, and mixed-precision handling. https://arxiv.org/abs/2304.11277
- **Shoeybi et al., 2019** — "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism." The reference for column- and row-parallel tensor parallelism. https://arxiv.org/abs/1909.08053
- **Huang et al., 2019** — "GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism." One of the two foundational pipeline-parallelism papers. https://arxiv.org/abs/1811.06965
- **Narayanan et al., 2019** — "PipeDream: Generalized Pipeline Parallelism for DNN Training." The other foundational PP paper; introduces 1F1B scheduling. https://arxiv.org/abs/1806.03377
- **Fedus et al., 2022** — "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity." The paper of record for large-scale MoE. https://arxiv.org/abs/2101.03961
- **Shazeer et al., 2017** — "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer." The original sparse-MoE paper cited by every MoE follow-up. https://arxiv.org/abs/1701.06538
- **Liu et al., 2023** — "Ring Attention with Blockwise Transformers for Near-Infinite Context." The long-context / sequence-parallelism reference. https://arxiv.org/abs/2310.01889

### Official tooling

- PyTorch FSDP API reference — https://pytorch.org/docs/stable/fsdp.html
- PyTorch FSDP2 (`fully_shard`) tutorial — https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html
- DeepSpeed features overview — https://www.deepspeed.ai/features/
- Hugging Face Accelerate "Concepts" (distributed types) — https://huggingface.co/docs/accelerate/concept_guides/big_model_inference

## Chapter 2 — Accelerate and PyTorch FSDP

### Papers

- **Zhao et al., 2023** — the FSDP paper (cited above); §3–4 cover the sharding, wrapping, and mixed-precision design.
- **Chowdhery et al., 2022** — "PaLM: Scaling Language Modeling with Pathways." Appendix on model / hardware FLOPS utilization — also relevant to chapter 5. https://arxiv.org/abs/2204.02311

### Official tooling

- Hugging Face Accelerate FSDP integration — https://huggingface.co/docs/accelerate/usage_guides/fsdp
- Accelerate config reference — https://huggingface.co/docs/accelerate/package_reference/cli
- Transformers `TrainingArguments` FSDP fields — https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments
- TRL distributed training guide — https://huggingface.co/docs/trl/customization
- PyTorch `torch.distributed.checkpoint` — https://pytorch.org/docs/stable/distributed.checkpoint.html
- PyTorch FSDP advanced guide (auto-wrap policies, mixed precision) — https://pytorch.org/tutorials/intermediate/FSDP_adavnced_tutorial.html

### Recipes

- Answer.AI, "You can now train a 70B model at home" — the reference write-up of the working QLoRA + FSDP recipe. https://www.answer.ai/posts/2024-03-06-fsdp-qlora.html
- Hugging Face `alignment-handbook` (Zephyr and related SFT/DPO recipes; ships working FSDP + DeepSpeed configs) — https://github.com/huggingface/alignment-handbook
- Axolotl (YAML-driven FSDP and DeepSpeed training) — https://github.com/axolotl-ai-cloud/axolotl

## Chapter 3 — DeepSpeed ZeRO-2/3 and CPU/NVMe offload

### Papers

- **Rajbhandari et al., 2020** — the ZeRO paper (cited above); the DeepSpeed-native description of the same accounting.
- **Ren et al., 2021** — "ZeRO-Offload: Democratizing Billion-Scale Model Training." The paper of record for CPU offload of optimizer state. https://arxiv.org/abs/2101.06840
- **Rajbhandari et al., 2021** — "ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning." The paper for NVMe offload and the trillion-parameter-on-a-single-DGX demonstrations. https://arxiv.org/abs/2104.07857
- **Wang et al., 2023** — "ZeRO++: Extremely Efficient Collective Communication for Giant Model Training." Optional; describes DeepSpeed's more recent communication-reduction techniques. https://arxiv.org/abs/2306.10209

### Official tooling

- DeepSpeed main docs — https://www.deepspeed.ai/
- DeepSpeed ZeRO configuration reference — https://www.deepspeed.ai/docs/config-json/#zero-optimizations-for-fp16-training
- Hugging Face Accelerate DeepSpeed integration — https://huggingface.co/docs/accelerate/usage_guides/deepspeed
- Transformers DeepSpeed integration guide — https://huggingface.co/docs/transformers/main_classes/deepspeed
- DeepSpeed `zero_to_fp32.py` checkpoint consolidator — https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/utils/zero_to_fp32.py
- DeepSpeed FLOPS profiler docs — https://www.deepspeed.ai/tutorials/flops-profiler/

## Chapter 4 — Memory, mixed precision, and checkpointing

### Papers

- **Chen et al., 2016** — "Training Deep Nets with Sublinear Memory Cost." The gradient / activation checkpointing paper. https://arxiv.org/abs/1604.06174
- **Micikevicius et al., 2018** — "Mixed Precision Training." The reference for FP16 mixed precision, loss scaling, and master weights in FP32. https://arxiv.org/abs/1710.03740
- **Micikevicius et al., 2022** — "FP8 Formats for Deep Learning." The definition of E4M3 / E5M2 and the training-scaling considerations. https://arxiv.org/abs/2209.05433
- **Dao et al., 2022** — "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." https://arxiv.org/abs/2205.14135
- **Dao, 2023** — "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning." https://arxiv.org/abs/2307.08691
- **Shah et al., 2024** — "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision." https://arxiv.org/abs/2407.08608
- **Dettmers et al., 2022** — "8-bit Optimizers via Block-wise Quantization." The `bitsandbytes` `AdamW8bit` paper. https://arxiv.org/abs/2110.02861
- **Jacobs et al., 2023** — "DeepSpeed Ulysses: System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models." The sequence-parallelism reference within DeepSpeed. https://arxiv.org/abs/2309.14509

### Official tooling

- NVIDIA Transformer Engine — https://github.com/NVIDIA/TransformerEngine
- Transformer Engine PyTorch API — https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/index.html
- FlashAttention (Dao-AILab) repository — https://github.com/Dao-AILab/flash-attention
- `torch.utils.checkpoint` reference — https://pytorch.org/docs/stable/checkpoint.html
- PyTorch `torch.distributed.algorithms._checkpoint.checkpoint_wrapper` — https://pytorch.org/docs/stable/distributed.algorithms.join.html
- `bitsandbytes` main repository — https://github.com/bitsandbytes-foundation/bitsandbytes
- Hugging Face `attn_implementation` guide — https://huggingface.co/docs/transformers/perf_infer_gpu_one
- PyTorch memory management (`PYTORCH_CUDA_ALLOC_CONF`) reference — https://pytorch.org/docs/stable/notes/cuda.html#memory-management

## Chapter 5 — Throughput, MFU, and profiling

### Papers

- **Chowdhery et al., 2022** — the PaLM paper (cited above); introduces MFU as the field's canonical efficiency metric.
- **Kaplan et al., 2020** — "Scaling Laws for Neural Language Models." Background for the `6 · N` per-token FLOPs approximation used in the MFU formula. https://arxiv.org/abs/2001.08361
- **Williams et al., 2009** — "Roofline: An Insightful Visual Performance Model for Multicore Architectures." The original roofline paper. https://dl.acm.org/doi/10.1145/1498765.1498785

### Official tooling

- PyTorch Profiler tutorial — https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
- PyTorch Profiler + TensorBoard plugin — https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html
- Perfetto trace viewer (drop-in replacement for `chrome://tracing`) — https://ui.perfetto.dev/
- DeepSpeed FLOPS profiler — https://www.deepspeed.ai/tutorials/flops-profiler/
- DeepSpeed autotuner (`wall_clock_breakdown`) — https://www.deepspeed.ai/tutorials/autotuning/
- NVIDIA Nsight Systems — https://developer.nvidia.com/nsight-systems
- `nvidia-smi` reference — https://docs.nvidia.com/deploy/nvidia-smi/
- `nvitop` (TUI wrapper for `nvidia-smi`) — https://github.com/XuehaiPan/nvitop

### Peak-FLOPS datasheets (verify current figures before quoting MFU)

- NVIDIA A100 datasheet — https://www.nvidia.com/en-us/data-center/a100/
- NVIDIA H100 datasheet — https://www.nvidia.com/en-us/data-center/h100/
- NVIDIA H200 datasheet — https://www.nvidia.com/en-us/data-center/h200/
- NVIDIA B100 / B200 (Blackwell) product pages — https://www.nvidia.com/en-us/data-center/

### Communication microbenchmark

- NVIDIA `nccl-tests` — https://github.com/NVIDIA/nccl-tests
- NCCL documentation — https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html

## Chapter 6 — Debugging OOM, NCCL, and load imbalance

### Papers

- **Goyal et al., 2017** — "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour." Referenced for the LR-scaling rules that come up when load-imbalance fixes materially change effective batch size. https://arxiv.org/abs/1706.02677

### Official tooling and references

- NCCL environment variables — https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html
- PyTorch distributed debugging (`TORCH_DISTRIBUTED_DEBUG`, `TORCH_NCCL_ASYNC_ERROR_HANDLING`, timeouts) — https://pytorch.org/docs/stable/distributed.html#debugging-torch-distributed-applications
- PyTorch memory snapshot / viewer — https://pytorch.org/docs/stable/torch_cuda_memory.html
- Interactive memory viewer at pytorch.org/memory_viz — https://pytorch.org/memory_viz
- `py-spy` — https://github.com/benfred/py-spy
- NVIDIA `nvidia-smi topo` (peer-access topology) — https://docs.nvidia.com/deploy/nvidia-smi/
- InfiniBand tooling reference (`ibv_devices`, `ibstat`) — https://docs.nvidia.com/networking/display/mlnxofedv461000/OpenFabrics+Alliance+(OFA)+Verbs+API

### Hugging Face-specific debugging guides

- Accelerate debugging tips — https://huggingface.co/docs/accelerate/concept_guides/deferring_execution
- Transformers `Trainer` distributed debugging — https://huggingface.co/docs/transformers/debugging

## Chapter 7 — Cluster sizing and handoff to `training-pipeline-engineer`

### Papers

- **Chowdhery et al., 2022** — the PaLM paper (cited above); Appendix on training-run economics and how their cluster shape was chosen.
- **Hoffmann et al., 2022** — "Training Compute-Optimal Large Language Models" (Chinchilla). Background for compute / token budgeting; useful when you're asked to justify how much data your fine-tune should see. https://arxiv.org/abs/2203.15556

### Cloud pricing and capacity references (verify current pricing before quoting)

- AWS EC2 GPU instance pricing — https://aws.amazon.com/ec2/instance-types/p5/
- Google Cloud GPU pricing — https://cloud.google.com/compute/gpus-pricing
- Azure NC / ND / NG series pricing — https://azure.microsoft.com/en-us/pricing/details/virtual-machines/linux/
- Lambda Labs on-demand and spot GPU pricing — https://lambdalabs.com/service/gpu-cloud
- CoreWeave GPU cloud — https://www.coreweave.com/pricing
- Together AI, Fireworks, Modal (managed training / inference alternatives to raw GPU rental) — check the current product pages for pricing

### Multi-node fabric references

- NVIDIA GPUDirect RDMA — https://docs.nvidia.com/cuda/gpudirect-rdma/
- NVIDIA NVLink and NVSwitch reference — https://www.nvidia.com/en-us/data-center/nvlink/
- InfiniBand HDR / NDR overviews — https://www.nvidia.com/en-us/networking/infiniband-adapters/

### Orchestration

- `torchrun` / `torch.distributed.run` — https://pytorch.org/docs/stable/elastic/run.html
- Slurm workload manager — https://slurm.schedmd.com/
- Kubernetes + Kubeflow Training Operator — https://www.kubeflow.org/docs/components/training/
- Ray (distributed training / Ray Train) — https://docs.ray.io/en/latest/train/train.html
- NVIDIA NeMo Framework Launcher — https://github.com/NVIDIA/NeMo-Framework-Launcher

## Reference training stacks

Full-stack projects worth reading end-to-end. They embody most of the choices this module discusses and each has its own opinions on the FSDP-vs-DeepSpeed axis.

- **NVIDIA NeMo Framework** (Megatron-Core-based; TP+PP+SP+EP first-class) — https://github.com/NVIDIA/NeMo
- **Meta llama-recipes** (FSDP-based Llama fine-tuning recipes) — https://github.com/meta-llama/llama-recipes
- **Hugging Face `alignment-handbook`** (FSDP + DeepSpeed SFT/DPO recipes for Zephyr and related) — https://github.com/huggingface/alignment-handbook
- **Axolotl** (YAML-driven training over FSDP / DeepSpeed / DDP) — https://github.com/axolotl-ai-cloud/axolotl
- **LLaMA-Factory** (all-in-one training framework covering full FT, LoRA, QLoRA under FSDP / DeepSpeed) — https://github.com/hiyouga/LLaMA-Factory
- **Megatron-DeepSpeed** (Microsoft's DeepSpeed integration on top of Megatron-LM; foundation-model scale) — https://github.com/microsoft/Megatron-DeepSpeed
- **Colossal-AI** (alternative distributed training framework with heterogeneous memory management) — https://github.com/hpcaitech/ColossalAI

## Suggested reading order for this module

1. Skim the ZeRO paper (Rajbhandari et al. 2020) §3, then read chapter 1 of this module. The memory table is the anchor for everything else.
2. Read chapter 2 with the PyTorch FSDP API reference open in another tab. Skim the FSDP paper (Zhao et al. 2023) §3–4 for the design rationale.
3. Read chapter 3 with the DeepSpeed config-json reference open. Skim the ZeRO-Offload paper (Ren et al. 2021) §3.
4. Read chapter 4 alongside the FlashAttention-2 paper's introduction (Dao 2023). Chapter 4 references numbers from all three FlashAttention papers; the FA2 intro is the shortest path to intuition.
5. Read chapter 5 with the PaLM appendix on MFU open. Compute MFU for one run of your own (exercise 03 Part B) before finishing the chapter.
6. Read chapter 6 last as a reference. It's most useful when you have a specific failure to look up; reading it cold is less rewarding than reading it while debugging.
7. Return to chapter 7 to size your next real run. The handoff table (fine-tuning-engineer vs. `training-pipeline-engineer`) is the section to bookmark.

<!-- needs-research: on the next authoring cycle, add version pins for the tooling links (torch, accelerate, transformers, deepspeed, trl, peft, flash-attn, bitsandbytes, transformer-engine) so future readers land on the version this module was authored against. Cloud pricing links and NVIDIA datasheet URLs should also be re-verified — both change with each product cycle. -->
