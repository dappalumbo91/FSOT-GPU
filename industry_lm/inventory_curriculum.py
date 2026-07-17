#!/usr/bin/env python3
"""
Inventory FSOT curriculum roots (D: training data + archive 2.1 + public data).

Does NOT train. Writes results/industry_lm/curriculum_inventory.json
so the solidification path is machine-checked before Phase 1+ curriculum.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "config" / "curriculum_roots.json"
OUT = ROOT / "results" / "industry_lm"
OUT.mkdir(parents=True, exist_ok=True)


def dir_summary(p: Path, max_children: int = 40) -> dict:
    if not p.exists():
        return {"path": str(p), "exists": False}
    children = []
    try:
        for c in sorted(p.iterdir(), key=lambda x: x.name.lower())[:max_children]:
            entry = {"name": c.name, "is_dir": c.is_dir()}
            if c.is_file():
                try:
                    entry["bytes"] = c.stat().st_size
                except OSError:
                    pass
            children.append(entry)
    except OSError as e:
        return {"path": str(p), "exists": True, "error": str(e)}
    n_files = n_dirs = 0
    total = 0
    try:
        for dirpath, dirnames, filenames in os.walk(p):
            n_dirs += len(dirnames)
            n_files += len(filenames)
            for f in filenames:
                try:
                    total += (Path(dirpath) / f).stat().st_size
                except OSError:
                    pass
            # cap walk cost on huge trees
            if n_files > 50000:
                break
    except OSError as e:
        return {
            "path": str(p),
            "exists": True,
            "children_sample": children,
            "walk_error": str(e),
        }
    return {
        "path": str(p),
        "exists": True,
        "n_files_walked": n_files,
        "n_dirs_walked": n_dirs,
        "bytes_walked": total,
        "gb_walked": round(total / (1024**3), 3),
        "children_sample": children,
    }


def main():
    cfg = json.loads(CFG.read_text(encoding="utf-8"))
    roots = {
        "training_data": Path(cfg["training_data"]),
        "lean_hub": Path(cfg["lean_hub"]),
        "public_data": Path(cfg["public_data"]),
        "archive_root": Path(cfg["archive_root"]),
    }
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": cfg,
        "roots": {},
        "lean_docs_sample": [],
        "public_domains_sample": [],
        "training_top_level": [],
        "ok": True,
    }
    for k, p in roots.items():
        print("scan", k, p)
        # lighter walk for huge training_data / public
        if k in ("training_data", "public_data"):
            if not p.exists():
                report["roots"][k] = {"path": str(p), "exists": False}
                report["ok"] = False
                continue
            kids = []
            for c in sorted(p.iterdir(), key=lambda x: x.name.lower()):
                kids.append({"name": c.name, "is_dir": c.is_dir()})
            report["roots"][k] = {
                "path": str(p),
                "exists": True,
                "top_level_count": len(kids),
                "top_level": kids[:60],
            }
            if k == "training_data":
                report["training_top_level"] = kids
            if k == "public_data":
                report["public_domains_sample"] = [x["name"] for x in kids if x["is_dir"]]
        else:
            report["roots"][k] = dir_summary(p, max_children=30)

    lean_docs = Path(cfg["lean_hub"]) / "docs"
    if lean_docs.is_dir():
        report["lean_docs_sample"] = sorted(
            [x.name for x in lean_docs.iterdir() if x.suffix in (".md", ".html", ".yaml")]
        )[:40]

    # key training files for FSOT core text
    td = Path(cfg["training_data"])
    for name in (
        "arxiv_fsot_core.txt",
        "DATA_CATALOG.md",
        "SOTA_PACKS.md",
        "questions_answers.csv",
    ):
        f = td / name
        report.setdefault("key_files", {})[name] = {
            "exists": f.is_file(),
            "bytes": f.stat().st_size if f.is_file() else None,
        }

    path = OUT / "curriculum_inventory.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("ok", report["ok"], "wrote", path)
    print("training top-level dirs", len(report.get("training_top_level") or []))
    print("public domains", len(report.get("public_domains_sample") or []))
    print("lean docs sample", len(report.get("lean_docs_sample") or []))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
