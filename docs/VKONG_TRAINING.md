# Train SpatialMQA LLaVA on vKong

This configuration is prepared for one A100, PyTorch 2.5.1/CUDA 12.1,
BF16, native SDPA and DeepSpeed ZeRO-2. It has not yet been tested on a rented GPU.
The configured hourly ceiling is $2; it is not a total spending limit.
Check available storage quota before starting: model caches, retained resumable
checkpoints and their final archive all consume space.

## Data and environment

The CLI sends scripts, requirements and six small annotation files only. It does
not upload local images or checkpoints. The workload copies the unchanged
annotations to `/data/spatial_mqa`, downloads the official 5,063 images from
`liuziyan/SpatialMQA` at a pinned revision, verifies image integrity and references,
and records annotation hashes. No S3 bucket or AWS credentials are needed.

Setup installs a pinned LLaVA source revision without its old dependency pins;
the training dependencies are supplied separately in requirements-vkong.txt.
This intentionally differs from upstream's torch 2.1.2 dependency declaration.
Setup tests a tiny LLaVA SDPA forward/backward on CUDA; this is not a full training
or dependency compatibility guarantee. No external flash-attn is installed.

## Run from the project root

After login and reviewing the workspace, resources and budget:

```bash
vkong validate
vkong run --detach --auto-stop
```

The default starts fresh if the stored output has no checkpoint. It does not
migrate checkpoints from the old server. When output already has checkpoints,
LLaVA automatically resumes the latest one; a preflight rejects obviously
incomplete/non-DeepSpeed state without deleting anything. The trainer performs
actual deserialization and compatibility checks. Keep the training settings
unchanged for resume.

The batch size is 2 with 8 accumulation steps (effective batch 16), for 10 epochs.
Follow logs using `vkong attach <instance-id>` with the ID returned by run.
No new rental is needed just to view logs.

## Retrieve results

Check compute release and successful storage save, even after auto-stop:

```bash
vkong app show spatial-mqa-llava
vkong storage show spatial-mqa-training
vkong storage files spatial-mqa-training checkpoints
vkong storage download spatial-mqa-training checkpoints/llava_1.5_7b_lora.tar.gz -o ./llava_1.5_7b_lora.tar.gz
```

The archive is produced only after successful training and contains retained
checkpoints as well as final adapter files and logs. Interrupted/failed jobs may
have individual checkpoints but no final archive. A directory's existence or a
stop acknowledgment does not prove a complete storage save.

For an intentional early stop:

```bash
vkong app stop spatial-mqa-llava
```

Storage is separately billable after compute stops. Local tests:

```bash
python -m unittest discover -s tests -p test_vkong_preflight.py -v
```
