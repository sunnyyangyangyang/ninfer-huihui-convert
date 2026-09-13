# ninfer-huihui-convert

Self-contained toolchain that converts
[huihui-ai/Huihui-Qwen3.8-27B-abliterated](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-27B-abliterated)
(HF BF16 safetensors, 18 shards, ~55.6 GiB) into **one** NInfer
`qwen3.8-27b/groupwise-int` `.ninfer` artifact (~16.9 GiB). Weights are stored
groupwise-int (Q4/Q5G64 + W8G32, FP16 group scales, `row-split-k128-v1` layout).

Only the download/quantize/assemble phases of the conversion touch tmpfs. At
most **two source shards** are resident at any moment, so the 55.6 GiB
checkpoint fits on a ~31 GiB tmpfs, and the final artifact + provenance report
land on durable storage under `out/`.

## Layout

| Path | What it is |
|---|---|
| `convert_huihui_ninfer.py` | conversion driver — the three-phase pipeline (below) |
| `tools/convert/` | conversion code vendored from [Neroued/ninfer](https://github.com/Neroued/ninfer) (Apache-2.0, master @ `a140e7a`), with two local patches applied |
| `tools/artifact/` | `.ninfer` container / layout / numeric-format code vendored from the same repo |
| `tools/freq_corpus/fixtures/ranking/` | token-frequency ranking data used to build the MTP draft head |
| `provenance-patch.diff` | the two local patches as a git diff (re-apply after upstream syncs) |
| `out/` | persistent artifact landing zone (gitignored; holds the `.ninfer` + `conversion.json` after a run) |

## Requirements

- Python >= 3.10 (verified on 3.13)
- `torch` (CUDA build if you have a GPU; a CPU fallback works), `safetensors`,
  `huggingface_hub`, `numpy` — e.g. `pip install torch safetensors huggingface_hub numpy`
- `/tmp` must be **tmpfs** with >= 25 GiB free (`MIN_FREE_GIB` is tunable in
  the driver). The pipeline is sized for a ~31 GiB tmpfs; peak budget is
  ~17 GiB payloads + <= 7 GiB shard window + ~3 GiB process.
- A network path to Hugging Face (anonymous, rate-limited, resumable);
  ~56 GiB of downloads.

## Usage

```bash
./convert_huihui_ninfer.py            # clears tmpfs automatically on completion; artifacts stay in out/
./convert_huihui_ninfer.py --keep     # leave the tmpfs working dirs for debugging
```

Interrupted? Re-run the same command: already-downloaded shards resume (hub
cache + LFS sha256 validation), and already-quantized object payloads are
skipped while they still exist on tmpfs (a reboot that clears tmpfs rebuilds
them).

## Three-phase pipeline (tmpfs strategy)

1. **phase 1** — download config/index + 6 frontend files into `/tmp`
   (HF cache on tmpfs), then run a light preflight: config-dimension
   validation, frontend resource name set, the 1124-object plan, and the MTP
   draft-head shortlist.
2. **phase 2** — for each of the 18 safetensors shards, in order: download it
   (resumable), validate the shape/dtype of every source tensor it holds from
   the safetensors header, quantize (on GPU) every object whose source tensors
   all live in the current shard window, write the encoded payloads to
   `/tmp`, then evict shards no remaining object needs. Resident shards <= 2.
3. **phase 3** — stream the payloads into a single `.ninfer` in object-plan
   order (each payload is deleted as it is written), finalize the container
   directory, compute sha256, write `conversion.json`, copy both to `out/`,
   and clear the tmpfs working dirs.

## Local patches (see `provenance-patch.diff`)

1. `tools/convert/qwen3_8_27b/convert.py::load_resources` — demote the
   frontend-resource SHA-256 mismatch from `raise` to `WARNING`. The checkpoint
   is a non-official (abliterated) variant, so its tokenizer resources differ
   from official Qwen3.8. All structural checks (config dimensions, source
   tensor names/shapes/dtypes, object plan) remain fully in force.
2. `tools/convert/common/safetensors.py::ShardReader` — `metadata()` reads the
   8-byte length prefix + JSON header directly instead of going through
   `safe_open`, which rejects files whose data span is not fully covered and
   would break validation of partially downloaded shards.

## Artifacts

- `out/qwen3_8_27b_huihui.ninfer` — ~16.9 GiB, 1124 objects (1118 tensors +
  6 frontend resources), identity `qwen3.8-27b/groupwise-int`, recipe
  `qwen3_8_27b-v1`
- `out/qwen3_8_27b_huihui.ninfer.conversion.json` — provenance report (source
  repo, sha256, device, elapsed time)

Verified run (2026-09-03): 18,210,531,328 bytes, sha256
`8c9f9d67a07ac97506978f6db6695d8074f78dec0fb80c4a85a8fb6fbedd7f03`.

## Running inference (optional)

The artifact is consumed by the NInfer engine from the upstream repo:

```bash
git clone https://github.com/Neroued/ninfer && cd ninfer
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release   # CUDA 13.1+, sm_120a
cmake --build build -j
./build/apps/ninfer /path/to/out/qwen3_8_27b_huihui.ninfer \
  --prompt "hi" --max-context 32768 --max-new 8192 \
  --kv-dtype fp8 --spec mtp --draft-tokens 3 --lm-head-draft
```

The artifact identity is the official `qwen3.8-27b/groupwise-int`; the runtime
binds by target key, and the weight content comes from the Huihui source.
Disabling the official-resource SHA check affects the conversion side only,
not the runtime.

## Vendoring note

`tools/` is vendored from Neroued/ninfer @ master (commit `a140e7a`) plus the
two local patches above. To re-sync with an upstream release: pull the
upstream tree, re-apply `provenance-patch.diff`, and copy back
`tools/convert`, `tools/artifact`, and the ranking data.
