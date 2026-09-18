#!/usr/bin/env python3
"""Slice-download conversion of huihui-ai/Huihui-Qwen3.8-27B-abliterated
(HF BF16 safetensors, 18 shards, ~56 GiB) into ONE NInfer v3 groupwise-int
.ninfer artifact (~17 GiB, qwen3.8-27b/groupwise-int family).

v3-era pipeline (NInfer master @ f76e19c0, 2026-09-17; vendored in tools/):

  phase 1 -- verify the NVMe scratch volume, download config/index + frontend
             resources (HF cache on the scratch volume; huggingface_hub,
             resumable, LFS sha256 validation), then run the full preflight:
             Qwen3.5-family config parsing, tokenizer token-domain validation,
             the official qwen3_8_27b recipe assignment (Q4/Q5 projections +
             Q8 embedding/output, Q6 vision patch embedding, Q8 mergers/MTP),
             and the indexed proposal head (131,072 rows from the vendored
             frequency ranking). The chat template is pinned to the maintained
             tools/chat_templates/qwen3_8.jinja via a resource override.
  phase 2 -- download ALL 18 safetensors shards, structurally validate every
             shard header (the v3 SafetensorsSource reads the 8-byte length
             prefix + JSON header directly, so validation is exact and works
             on the complete files), then run the v3 converter in ONE
             streaming pass: ArtifactWriter receives the complete precomputed
             object plan up front and each weight job's quantized rows
             (groupwise int: Q4/Q5G64 + Q8G32, FP16 group scales,
             row-split-k128-v1 layout; amax/encode on GPU) stream straight
             into the final file -- no intermediate payload directory.
  phase 3 -- the converter's conversion.json report (source provenance +
             artifact sha256 added), artifact + report copied to the repo's
             out/, scratch volume cleared (unless --keep).

Why NVMe scratch instead of the v2-era tmpfs shard window: the v3 converter's
prepare() step touches every source tensor (a one-element preflight per
input) before producing anything, so the old "at most two shards resident"
window no longer fits the converter; the ~56 GiB of shards fits comfortably
on the scratch volume, and the v3 streaming writer removes the need for the
payload staging directory entirely.

No local patches are required anymore: v2's two provenance patches are
absorbed or obsolete upstream -- direct safetensors header reads are now
native to the v3 SafetensorsSource, and the v3 resource loader performs no
SHA-256 provenance check at all (the maintained chat template is supplied
via the resource override instead).

Run from anywhere:  ./convert_huihui_ninfer.py [--keep]
  (default: after conversion the artifact + report are copied to ./out/ and
   the scratch volume is cleared; --keep leaves it in place)
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent  # vendored tools/ lives at the repo root
PERSISTENT_OUT = REPO_ROOT / "out"
sys_path_boot = __import__("sys")
sys_path_boot.path.insert(0, str(REPO_ROOT))

MODEL_NAME = "huihui-ai/Huihui-Qwen3.8-27B-abliterated"
SCRATCH = REPO_ROOT / "scratch"          # NVMe working volume (HF cache + shards live here)
HF_HOME = SCRATCH / "hf-cache"           # huggingface_hub cache root (on NVMe, not tmpfs)
ARTIFACT_OUT = SCRATCH / "qwen3_8_27b_huihui.ninfer"
REPORT_OUT = Path(str(ARTIFACT_OUT) + ".conversion.json")

# Frontend resources pulled from the Huihui checkpoint; chat_template.jinja is
# deliberately NOT among them -- the maintained official template is pinned
# via a resource override (see CHAT_TEMPLATE).
FRONTEND_FILES = (
    "config.json",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
)

CHAT_TEMPLATE = REPO_ROOT / "tools" / "chat_templates" / "qwen3_8.jinja"
RANKING = REPO_ROOT / "tools" / "freq_corpus" / "fixtures" / "ranking" / "ranking.train.counts.i64"
COMPONENTS = ("text", "vision", "mtp")   # dflash2 is not present in the Huihui base checkpoint
PROPOSAL_ROWS = 131072

# Scratch volume budget: ~56 GiB shards + ~17 GiB artifact + headroom.
MIN_FREE_GIB = 80

ALLOWED_SOURCE_DTYPES = {"BF16", "F16", "F32", "I32", "I64", "I8", "U8", "F8_E4M3"}

MODEL_DIR: Path = Path(".")  # set to the hub snapshot dir by phase1


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def scratch_used_mb() -> int:
    return shutil.disk_usage(str(SCRATCH)).used // 1024**2


def scratch_check() -> None:
    """Assert the scratch volume is real disk (not tmpfs) with headroom."""
    SCRATCH.mkdir(parents=True, exist_ok=True)
    target = str(SCRATCH.resolve())
    fstype = "unknown"
    best = ""
    with open("/proc/mounts", encoding="ascii", errors="replace") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 3:
                continue
            mount = parts[1]
            if (target + "/").startswith(mount + "/") or mount == target:
                if len(mount) > len(best):
                    best = mount
                    fstype = parts[2]
    if fstype == "tmpfs":
        raise RuntimeError(
            f"scratch volume {target} is tmpfs; the v3 pipeline needs ~80 GiB of "
            "durable scratch (shards + artifact) -- point the repo at NVMe space"
        )
    free = shutil.disk_usage(str(SCRATCH)).free
    if free < MIN_FREE_GIB * 1024**3:
        raise RuntimeError(
            f"scratch free {free // 1024**3} GiB < required {MIN_FREE_GIB} GiB; "
            "clear the scratch volume or lower MIN_FREE_GIB"
        )
    log(f"scratch check ok: {target} is {fstype}, free {free // 1024**3} GiB")


def sha256_of(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(16 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(filename: str) -> Path:
    """One file through huggingface_hub (resumes in-cache, validates LFS sha256)."""
    from huggingface_hub import hf_hub_download

    log(f"downloading {filename} via hf-hub [scratch {scratch_used_mb()} MiB]")
    started = time.time()
    r = hf_hub_download(MODEL_NAME, filename, cache_dir=str(HF_HOME))
    p = Path(getattr(r, "path", r))
    log(f"  {filename} done in {time.time() - started:.0f}s -> {p}")
    return p


def shard_order(model_dir: Path) -> list[str]:
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())
    import re

    shards = sorted(
        set(index["weight_map"].values()),
        key=lambda s: int(re.search(r"model-(\d+)-of", s).group(1)),
    )
    return shards


def preflight(model_dir: Path) -> None:
    """Config parsing, resource/token-domain validation, recipe + proposal build.

    Nothing here touches shard data, so it can run before the shards download:
    build_model only needs config.json + frontend resources; the recipe and
    proposal head construction are pure config work (source reads stay lazy
    until the v3 prepare() step in phase 2).
    """
    from tools.convert.official_recipes import RECIPES
    from tools.convert.proposal import add_official_proposal
    from tools.convert.qwen3_5 import build_model
    from tools.convert.recipe import Recipe
    from tools.convert.sources.safetensors import SafetensorsSource

    if not CHAT_TEMPLATE.is_file():
        raise RuntimeError(f"maintained chat template missing: {CHAT_TEMPLATE}")
    if not RANKING.is_file():
        raise RuntimeError(f"proposal ranking fixture missing: {RANKING}")

    with SafetensorsSource(str(model_dir)) as base:
        model = build_model(
            base,
            components=COMPONENTS,
            resource_overrides={"chat_template.jinja": str(CHAT_TEMPLATE)},
        )
        recipe = Recipe(model)
        RECIPES["qwen3_8_27b"](model, recipe, {"base": base})
        add_official_proposal(recipe, ranking=str(RANKING), rows=PROPOSAL_ROWS)
    log(
        "preflight OK: "
        f"{len(model.parameters)} logical parameters, "
        f"vocab {model.config['vocab_size']}, "
        f"{len(model.components)} components {sorted(model.components)}, "
        f"token domain {model.token_count}, proposal {PROPOSAL_ROWS} rows"
    )


def phase1() -> None:
    global MODEL_DIR
    scratch_check()
    HF_HOME.mkdir(parents=True, exist_ok=True)
    first = download_file(FRONTEND_FILES[0])
    MODEL_DIR = first.parent  # hub snapshot dir: all repo files land here
    log(f"model dir = {MODEL_DIR}")
    for name in FRONTEND_FILES[1:]:
        download_file(name)
    log("running preflight (config dims, resources/token domain, recipe, proposal)...")
    preflight(MODEL_DIR)


def validate_shards() -> None:
    """Per-shard structural validation straight from the safetensors headers."""
    from tools.convert.sources.safetensors import SafetensorsSource

    with SafetensorsSource(str(MODEL_DIR)) as store:
        by_shard: dict[str, list[str]] = {}
        for name, file in store.weight_map.items():
            by_shard.setdefault(str(file), []).append(name)
        for shard in shard_order(MODEL_DIR):
            file = MODEL_DIR / shard
            if not file.is_file():
                raise RuntimeError(f"shard missing after download: {file}")
            infos = store._header(file)  # direct 8-byte-prefix + JSON header read
            for name in by_shard[str(file)]:
                info = infos[name]
                if info.dtype not in ALLOWED_SOURCE_DTYPES:
                    raise ValueError(f"{shard}/{name}: unsupported source dtype {info.dtype}")
            log(f"  {shard}: header validated ({len(infos)} tensors)")


def convert_v3() -> dict:
    """Single streaming pass of the v3 converter over the downloaded shards."""
    from tools.convert.official_recipes import RECIPES
    from tools.convert.pipeline import convert
    from tools.convert.proposal import add_official_proposal
    from tools.convert.qwen3_5 import build_model
    from tools.convert.quantization.groupwise import pick_device
    from tools.convert.recipe import Recipe
    from tools.convert.sources.safetensors import SafetensorsSource

    device = pick_device("cuda")
    log(f"quantization device: {device}")
    if ARTIFACT_OUT.exists() or REPORT_OUT.exists():
        log("removing stale partial artifact/report from an interrupted run")
        ARTIFACT_OUT.unlink(missing_ok=True)
        REPORT_OUT.unlink(missing_ok=True)

    started = time.time()

    def progress(index: int, total: int, job) -> None:
        label = job.parameters[0]
        if len(job.parameters) > 1:
            label += f" (+{len(job.parameters) - 1})"
        log(
            f"  [{index + 1}/{total}] {label}: {job.spec.format} {job.spec.shape} "
            f"[scratch {scratch_used_mb()} MiB]"
        )

    with SafetensorsSource(str(MODEL_DIR)) as base:
        model = build_model(
            base,
            components=COMPONENTS,
            resource_overrides={"chat_template.jinja": str(CHAT_TEMPLATE)},
        )
        recipe = Recipe(model)
        RECIPES["qwen3_8_27b"](model, recipe, {"base": base})
        add_official_proposal(recipe, ranking=str(RANKING), rows=PROPOSAL_ROWS)
        report = convert(
            model,
            recipe,
            ARTIFACT_OUT,
            name="qwen3.8-27b",
            provenance={
                "converter": "ninfer-v3",
                "recipe": "qwen3_8_27b (official) + indexed proposal head",
                "sources": {
                    "base": {
                        "repo": MODEL_NAME,
                        "path": str(MODEL_DIR),
                        "files": "HF safetensors shards (18) + config + frontend resources",
                    }
                },
                "ranking": str(RANKING),
                "provenance_note": (
                    "converted from the non-official Huihui abliterated checkpoint; "
                    "weights are Huihui's, structure matches the official "
                    "qwen3.8-27b/groupwise-int family; the maintained official "
                    "chat template is pinned via resource override; source "
                    "header shape/dtype validated per shard during phase 2"
                ),
            },
            device=str(device),
            rows_per_chunk=512,
            progress=progress,
        )
    elapsed = time.time() - started
    report["source_sha256_note"] = "sha256 of the artifact file is recorded below"
    report["artifact_sha256"] = sha256_of(ARTIFACT_OUT)
    report_path = Path(str(ARTIFACT_OUT) + ".conversion.json")
    data = json.loads(report_path.read_text())
    data["source_sha256_note"] = report["source_sha256_note"]
    data["artifact_sha256"] = report["artifact_sha256"]
    report_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    log(f"conversion done: {report['objects']} objects in {elapsed:.0f}s")
    log(f"artifact: {ARTIFACT_OUT.stat().st_size / 1024**3:.2f} GiB")
    log(f"sha256: {report['artifact_sha256']}")
    return report


def phase2() -> dict:
    for shard in shard_order(MODEL_DIR):
        if not (MODEL_DIR / shard).is_file():
            download_file(shard)
            continue
        log(f"  {shard} already present (resume)")
    validate_shards()
    log("phase 2: single-pass v3 conversion (prepare + streaming quantization)...")
    return convert_v3()


def final_collect(keep: bool = False) -> None:
    """Copy artifact + report to the persistent out/ dir, then clear scratch."""
    if not ARTIFACT_OUT.exists():
        log("nothing to collect (artifact missing)")
        return
    PERSISTENT_OUT.mkdir(parents=True, exist_ok=True)
    dest = PERSISTENT_OUT / ARTIFACT_OUT.name
    shutil.copy2(ARTIFACT_OUT, dest)
    if dest.stat().st_size != ARTIFACT_OUT.stat().st_size:
        raise RuntimeError("artifact copy size mismatch")
    if REPORT_OUT.exists():
        shutil.copy2(REPORT_OUT, PERSISTENT_OUT / REPORT_OUT.name)
    log(f"persistent copy: {dest}")
    if keep:
        log("--keep set: scratch volume left in place")
        return
    shutil.rmtree(SCRATCH, ignore_errors=True)
    log(f"scratch cleared; final size {scratch_used_mb()} MiB")


def main() -> None:
    parser = __import__("argparse").ArgumentParser(description=__doc__)
    parser.add_argument("--keep", action="store_true", help="leave scratch after conversion")
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")          # xet breaks on some CDN edges
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    t0 = time.time()
    phase1()
    phase2()
    final_collect(keep=args.keep)
    log(f"ALL DONE in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
