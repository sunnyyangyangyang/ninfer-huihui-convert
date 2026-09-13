#!/usr/bin/env python3
"""Slice-streaming conversion of huihui-ai/Huihui-Qwen3.8-27B-abliterated
(HF BF16 safetensors) into ONE NInfer groupwise-int .ninfer artifact.

Download mechanism (huggingface_hub based, proven in production runs):
  - huggingface_hub (NOT raw curl) with HF_HUB_DISABLE_XET=1 (xet breaks on
    some CDN edges);
  - the whole HF cache lives on tmpfs (HF_HOME under /tmp);
  - tmpfs type assertion + free-space gate before starting.

Memory strategy (tmpfs-hosted):
  phase 1: download frontend files + config + index, run a light preflight
           (config dims, resource names/hashes, object plan, draft shortlist).
  phase 2: for each shard in order: hub-download it, validate its tensors'
           shape/dtype from the safetensors header, quantize every object whose
           source tensors all live in the current shard window, store encoded
           payloads on tmpfs, then delete shards (symlink + blob) no remaining
           object needs. Only 1-2 shards are resident at a time.
  phase 3: assemble the .ninfer from the stored payloads in plan order,
           deleting each payload as it is written (streaming copy).

Run from anywhere:  ./convert_huihui_ninfer.py [--keep]
  (default: after assembly the artifact + report are copied to ./out/ and all
   tmpfs working dirs are deleted; --keep leaves them in place)
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent  # vendored tools/ lives at the repo root
PERSISTENT_OUT = Path(__file__).resolve().parent / "out"
sys_path_boot = __import__("sys")
sys_path_boot.path.insert(0, str(REPO_ROOT))

MODEL_NAME = "huihui-ai/Huihui-Qwen3.8-27B-abliterated"
HF_HOME = Path("/tmp/huihui-hf")          # huggingface_hub cache root (tmpfs)
PAYLOAD_DIR = Path("/tmp/huihui-payloads")
ARTIFACT_OUT = Path("/tmp/qwen3_8_27b_huihui.ninfer")
REPORT_OUT = Path(str(ARTIFACT_OUT) + ".conversion.json")

FRONTEND_FILES = (
    "config.json",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "generation_config.json",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
)

MIN_FREE_GIB = 25  # tmpfs free-space gate before starting


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def tmpfs_used_mb() -> int:
    return shutil.disk_usage("/tmp").used // 1024**2


def check_tmpfs() -> None:
    """Assert /tmp really is tmpfs and has headroom before starting."""
    fstype = "unknown"
    with open("/proc/mounts", encoding="ascii", errors="replace") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) >= 3 and parts[1] == "/tmp":
                fstype = parts[2]
                break
    if fstype != "tmpfs":
        raise RuntimeError(
            f"/tmp is '{fstype}', not tmpfs; refusing (this pipeline is sized for tmpfs)"
        )
    free = shutil.disk_usage("/tmp").free
    if free < MIN_FREE_GIB * 1024**3:
        raise RuntimeError(
            f"/tmp free {free // 1024**3} GiB < required {MIN_FREE_GIB} GiB; "
            "clear tmpfs or lower MIN_FREE_GIB"
        )
    log(f"tmpfs check ok: /tmp is tmpfs, free {free // 1024**3} GiB")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(16 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(filename: str) -> Path:
    """One file through huggingface_hub (resumes in-cache, validates LFS sha256)."""
    from huggingface_hub import hf_hub_download

    log(f"downloading {filename} via hf-hub [tmpfs {tmpfs_used_mb()} MiB]")
    started = time.time()
    r = hf_hub_download(MODEL_NAME, filename, cache_dir=str(HF_HOME))
    p = Path(getattr(r, "path", r))
    log(f"  {filename} done in {time.time() - started:.0f}s -> {p}")
    return p


def evict_file(filename: str) -> None:
    """Remove a shard from the hub cache: the snapshot symlink and its blob."""
    link = MODEL_DIR / filename
    if not (link.exists() or link.is_symlink()):
        return
    try:
        blob = link.resolve()
        if str(blob).startswith(str(HF_HOME)) and blob.exists() and not blob.is_symlink():
            blob.unlink()
    finally:
        link.unlink(missing_ok=True)
    log(f"  evicted {filename} [tmpfs {tmpfs_used_mb()} MiB]")


MODEL_DIR: Path = Path(".")  # set to the hub snapshot dir by phase1


def shard_order(model_dir: Path) -> list[str]:
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())
    shards = sorted(
        set(index["weight_map"].values()),
        key=lambda s: int(re.search(r"model-(\d+)-of", s).group(1)),
    )
    return shards


def preflight_light(model_dir: Path):
    """Config/resources/plan/draft preflight without source-tensor checks.

    Source shape/dtype validation happens per shard in phase 2 (source
    metadata is read lazily from the safetensors headers, never up front)."""
    from tools.convert.qwen3_6.common.conversion import load_json
    from tools.convert.qwen3_6.common.recipe import SourcePreflight, source_requirements
    from tools.convert.qwen3_6_27b import convert as qwen3_6_convert
    from tools.convert.qwen3_6_27b import draft_head, recipe
    from tools.convert.qwen3_8_27b import convert as conv

    model = Path(model_dir)
    config_summary = qwen3_6_convert.validate_config(load_json(model / "config.json"))
    conv.preflight_inventory()
    source = SourcePreflight(
        recipe_count=len(recipe.RECIPE_SPECS),
        source_tensor_count=len(source_requirements(recipe.RECIPE_SPECS)),
        source_shard_count=len(set(shard_order(model))),
        source_dtype_counts={},
    )
    resources = conv.load_resources(model)
    object_plan = conv.build_object_plan({r.name: r.data for r in resources})
    draft = draft_head.compute_shortlist(conv._repo_root() / draft_head.DEFAULT_RANKING, model)
    return conv.ConversionPreflight(
        model_dir=model,
        config_summary=config_summary,
        source=source,
        resources=resources,
        draft=draft,
        object_plan=object_plan,
    )


def phase1():
    global MODEL_DIR
    HF_HOME.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    first = download_file(FRONTEND_FILES[0])
    MODEL_DIR = first.parent  # hub snapshot dir: all repo files land here
    log(f"model dir = {MODEL_DIR}")
    for name in FRONTEND_FILES[1:]:
        download_file(name)
    log("running light preflight (config dims, resource names/hashes, object plan, draft shortlist)...")
    pre = preflight_light(MODEL_DIR)
    log("preflight OK (source shape/dtype checks deferred to per-shard phase 2)")
    return pre


def phase2() -> None:
    import torch

    from tools.convert.common.quantize import pick_device
    from tools.convert.common.safetensors import ShardReader
    from tools.convert.qwen3_6.common.recipe import source_requirements
    from tools.convert.qwen3_6_27b import convert as qwen3_6_convert
    from tools.convert.qwen3_6_27b import recipe
    from tools.convert.qwen3_8_27b import inventory

    device = pick_device("cuda")
    log(f"quantization device: {device}")
    PAYLOAD_DIR.mkdir(parents=True, exist_ok=True)

    weight_map = json.loads((MODEL_DIR / "model.safetensors.index.json").read_text())["weight_map"]
    reqs = source_requirements(recipe.RECIPE_SPECS)
    reqs_by_shard: dict[str, dict] = {}
    for name, req in reqs.items():
        reqs_by_shard.setdefault(weight_map[name], {})[name] = req

    def obj_shards(spec) -> frozenset[str]:
        if isinstance(spec, inventory.ResourceSpec):
            return frozenset()
        rec = recipe.RECIPES_BY_NAME[spec.name]
        from tools.convert.qwen3_6.common.recipe import source_requirements as sr

        return frozenset(weight_map[s] for s in sr([rec]))

    all_specs = list(inventory.OBJECT_SPECS)
    shards_needed = {s.name: obj_shards(s) for s in all_specs}
    shard_names = shard_order(MODEL_DIR)

    def payload_path(spec) -> Path:
        idx = all_specs.index(spec)
        safe = spec.name.replace("/", "_").replace(":", "_")
        return PAYLOAD_DIR / f"obj-{idx:04d}-{safe}.bin"

    pending = [
        s for s in all_specs
        if not isinstance(s, inventory.ResourceSpec) and not payload_path(s).exists()
    ]
    tensor_total = sum(1 for s in all_specs if not isinstance(s, inventory.ResourceSpec))
    completed = tensor_total - len(pending)
    log(f"phase 2: {len(pending)} tensor objects to materialize+quantize "
        f"across {len(shard_names)} shards")
    window: set[str] = set()

    for shard in shard_names:
        if shard not in window:
            download_file(shard)
            window.add(shard)
            # per-shard structural validation from the (full) safetensors header
            if shard in reqs_by_shard:
                with ShardReader(MODEL_DIR) as reader:
                    meta = reader.metadata(list(reqs_by_shard[shard]))
                for name, req in reqs_by_shard[shard].items():
                    actual = meta[name]
                    if actual.shape != req.shape or actual.dtype != req.dtype:
                        raise ValueError(
                            f"{name}: source shape {actual.shape} dtype {actual.dtype} "
                            f"!= required {req.shape} {req.dtype}"
                        )
                log(f"  {shard}: header validated ({len(reqs_by_shard[shard])} tensors)")
        with ShardReader(MODEL_DIR) as reader:
            for spec in list(pending):
                if shards_needed[spec.name] - window:
                    continue
                started = time.time()
                tensor = qwen3_6_convert.materialize_tensor(spec, reader, DRAFT_CTX[0])
                payload = qwen3_6_convert.encode_tensor_payload(tensor, spec, device)
                del tensor
                payload_path(spec).write_bytes(payload)
                del payload
                completed += 1
                pending.remove(spec)
                torch.cuda.empty_cache() if device.type == "cuda" else None
                log(
                    f"  [{completed}/{tensor_total}] {spec.name} (shard {shard}) "
                    f"in {time.time() - started:.1f}s [tmpfs {tmpfs_used_mb()} MiB]"
                )
        needed: set[str] = set()
        for spec in pending:
            needed |= shards_needed[spec.name]
        for old in sorted(window - needed):
            evict_file(old)
            window.discard(old)
        if not pending:
            break
        log(f"shard {shard} window done; resident={sorted(window)} [tmpfs {tmpfs_used_mb()} MiB]")
    if pending:
        raise RuntimeError(f"shards exhausted with pending objects: {[s.name for s in pending]}")
    log("phase 2 complete: all tensor payloads on tmpfs")


DRAFT_CTX: tuple = ()


def phase3_assemble() -> None:
    from tools.artifact.container import ArtifactIdentity, ArtifactWriter
    from tools.convert.qwen3_8_27b import inventory
    from tools.convert.qwen3_8_27b.convert import RECIPE_ID

    preflight = preflight_light(MODEL_DIR)  # re-verify + resource bytes
    all_specs = list(inventory.OBJECT_SPECS)
    resources = {r.name: r.data for r in preflight.resources}
    log(f"phase 3: assembling {ARTIFACT_OUT}")
    started = time.time()
    writer = ArtifactWriter(
        ARTIFACT_OUT,
        ArtifactIdentity(inventory.MODEL_ID, inventory.WEIGHTS_ID),
        preflight.object_plan.specs,
    )
    try:
        for idx, spec in enumerate(all_specs):
            if isinstance(spec, inventory.ResourceSpec):
                writer.write(spec.name, resources[spec.name])
            else:
                p = payload_path_global(idx, spec)
                if not p.exists():
                    raise RuntimeError(f"missing payload {p}")
                with p.open("rb") as fh:
                    writer.write(spec.name, fh)
                p.unlink()
        writer.finish()
    finally:
        writer.close()
    final_bytes = ARTIFACT_OUT.stat().st_size
    elapsed = time.time() - started
    report = {
        "source": {
            "repo": MODEL_NAME,
            "files": "HF safetensors shards (18) + config + 6 frontend resources",
        },
        "identity": {
            "model_id": inventory.MODEL_ID,
            "weights_id": inventory.WEIGHTS_ID,
            "target_key": inventory.TARGET_KEY,
            "recipe_id": RECIPE_ID,
        },
        "out": str(ARTIFACT_OUT),
        "bytes": final_bytes,
        "sha256": sha256_of(ARTIFACT_OUT),
        "objects": len(preflight.object_plan.objects),
        "payload_span_bytes": preflight.object_plan.payload_span_bytes,
        "elapsed_seconds": round(elapsed, 1),
        "device": "cuda" if __import__("torch").cuda.is_available() else "cpu",
        "provenance_note": (
            "converted from a non-official source; official Qwen3.8 resource "
            "SHA-256 provenance check was disabled by local patch; source "
            "shape/dtype validated per shard during phase 2"
        ),
    }
    REPORT_OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    log(f"artifact: {final_bytes / 1024**3:.2f} GiB in {elapsed:.1f}s")
    log(f"sha256: {report['sha256']}")
    log(f"report: {REPORT_OUT}")


def payload_path_global(idx: int, spec) -> Path:
    safe = spec.name.replace("/", "_").replace(":", "_")
    return PAYLOAD_DIR / f"obj-{idx:04d}-{safe}.bin"


def final_collect(keep: bool = False) -> None:
    """Copy artifact + report to the persistent out/ dir, then clear tmpfs."""
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
        log("--keep set: tmpfs working dirs left in place")
        return
    for path in (PAYLOAD_DIR, HF_HOME):
        shutil.rmtree(path, ignore_errors=True)
    ARTIFACT_OUT.unlink(missing_ok=True)
    REPORT_OUT.unlink(missing_ok=True)
    log(f"tmpfs cleared; final size {tmpfs_used_mb()} MiB")


def main() -> None:
    global DRAFT_CTX
    import argparse as _ap

    parser = _ap.ArgumentParser(description=__doc__)
    parser.add_argument("--keep", action="store_true",
                        help="leave tmpfs working dirs after conversion")
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")          # xet breaks on some CDN edges
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    t0 = time.time()
    check_tmpfs()
    pre = phase1()
    DRAFT_CTX = (pre.draft,)
    phase2()
    phase3_assemble()
    final_collect(keep=args.keep)
    log(f"ALL DONE in {(time.time() - t0) / 60:.1f} min; tmpfs used {tmpfs_used_mb()} MiB")


if __name__ == "__main__":
    main()
