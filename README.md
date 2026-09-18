# ninfer-huihui-convert

Self-contained toolchain that converts
[huihui-ai/Huihui-Qwen3.8-27B-abliterated](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-27B-abliterated)
(HF BF16 safetensors, 18 shards, ~56 GiB) into **one** NInfer
**v3** `.ninfer` artifact (~17 GiB, groupwise-int: Q4/Q5G64 + Q8G32, FP16
group scales, `row-split-k128-v1` layout, indexed proposal head 131072 rows).

Vendored from [Neroued/ninfer](https://github.com/Neroued/ninfer) (Apache-2.0,
master @ `f76e19c0`, 2026-09-17) -- the v3 converter generation. No local
patches are required anymore: v2's two provenance patches are absorbed or
obsolete upstream (direct safetensors header reads are native to the v3
`SafetensorsSource`; the v3 resource loader performs no SHA-256 provenance
check, and the maintained official chat template is pinned via a resource
override instead).

## Layout

| Path | What it is |
|---|---|
| `convert_huihui_ninfer.py` | conversion driver -- three-phase pipeline (below) |
| `tools/convert/` | v3 conversion code vendored from Neroued/ninfer (pipeline, recipes, methods, sources) |
| `tools/artifact/` | v3 `.ninfer` container / layout / numeric-format code (reader, writer, codecs) |
| `tools/freq_corpus/fixtures/ranking/` | token-frequency ranking fixture used to build the proposal head |
| `tools/upgrade_ninfer_v2_to_v3.py` + `tools/chat_templates/` | offline v2->v3 upgrade tool (stdlib-only) + maintained Qwen chat templates |
| `out/` | persistent artifact landing zone (gitignored; `.ninfer` + `.conversion.json`) |
| `scratch/` | working volume (gitignored): HF cache + shards during a run, cleared afterwards |

## Requirements

- Python >= 3.10 (verified on 3.13), `torch` (CUDA build if you have a GPU;
  CPU fallback works), `safetensors`, `huggingface_hub`, `numpy`
- An **NVMe-class scratch volume** with >= 80 GiB free (`MIN_FREE_GIB` is
  tunable in the driver): all 18 shards plus the final artifact live on it
  during the run. The v2-era tmpfs shard-window trick no longer applies,
  because the v3 converter's prepare() step touches every source tensor
  before producing; 56 GiB of shards fit comfortably on durable storage, and
  the v3 streaming writer streams each quantized object straight into the
  final file (no payload staging directory).
- A network path to Hugging Face (anonymous, rate-limited, resumable);
  ~56 GiB of downloads.

## Usage

```bash
./convert_huihui_ninfer.py            # clears the scratch volume automatically; artifacts stay in out/
./convert_huihui_ninfer.py --keep     # leave the scratch volume in place for debugging
```

Interrupted? Re-run the same command: already-downloaded shards resume from
the HF cache on the scratch volume (LFS sha256 validation); a partial
artifact from an interrupted run is removed and re-quantized (quantization
itself is a single streaming pass, ~10-20 min on a 5090).

### Phases

1. **phase 1** -- verify the scratch volume, download config/index + 6
   frontend resources, then the full preflight: Qwen3.5-family config
   parsing, tokenizer token-domain validation, the official `qwen3_8_27b`
   recipe assignment (Q4/Q5 projections, Q8 embedding/output, Q6/Q8/Q4-Q5
   vision, Q8 MTP), and the indexed proposal head (131,072 rows from the
   vendored ranking). The chat template is pinned to the maintained
   `tools/chat_templates/qwen3_8.jinja`.
2. **phase 2** -- download all 18 shards (resumable), structurally validate
   every shard header from the safetensors files (direct 8-byte prefix +
   JSON header read), then run the v3 converter in one streaming pass;
   quantization runs on the GPU (rows_per_chunk=512).
3. **phase 3** -- the converter's `conversion.json` report (with source
   provenance + artifact sha256 added) and the artifact are copied to
   `out/`; the scratch volume is cleared.

## Upgrading an existing v2 artifact (no re-conversion)

The engine now requires v3 artifacts; v2 files are rejected with a pointer
to the offline upgrade tool. If you already have a v2 `.ninfer` built with
this checkpoint (or any official Qwen3.6/3.8-27B groupwise-int / NVFP4
artifact), upgrade it in place -- weight bytes are preserved, the maintained
chat template is installed, no re-download or re-quantization:

```bash
python3 tools/upgrade_ninfer_v2_to_v3.py INPUT.ninfer OUTPUT.ninfer
```

## Artifacts

- `out/qwen3_8_27b_huihui.ninfer` -- v3 container, ~17 GiB, 1124 objects
  (1118 tensors + 6 frontend resources), 1422 logical bindings, components
  text+vision+mtp, proposal head 131072 rows, identity
  `qwen3.8-27b/groupwise-int` family
- `out/qwen3_8_27b_huihui.ninfer.conversion.json` -- provenance report
  (source repo, recipe, devices, per-job formats, artifact id, sha256)

Weights are Huihui's (abliterated); structure and numeric formats match the
official qwen3.8-27b groupwise-int family, so the runtime binds exactly like
the official model. The DFlash2 speculative backend is not present in the
Huihui base checkpoint, so MTP remains the speculative decoding path.
