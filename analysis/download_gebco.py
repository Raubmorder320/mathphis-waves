"""Download the official GEBCO 2026 regional subset, including source-type IDs.

Run: python -m analysis.download_gebco
No account or email is supplied. Raw files and request provenance stay together.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
import urllib.request
import zipfile
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
BASE = "https://download.gebco.net"


def request_json(path: str, payload: dict | None = None):
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(BASE + path, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as response:
        return json.load(response)


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    manifest_path = RAW / "gebco_request.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        grids = request_json("/api/grids")
        grid = next(g for g in grids if g["name"] == "gebco_2026_global")
        source_ids = [s["id"] for s in grid["data_sources"]
                      if s["name"] in {"gebco_2026", "gebco_2026_tid"}]
        assert len(source_ids) == 2
        formats = request_json("/api/formats")
        netcdf_id = next(f["id"] for f in formats if f["name"] == "netcdf")
        payload = {
            "id": "0", "email": None,
            "submission_date": datetime.now(timezone.utc).isoformat(),
            "processing_status": "new",
            "items": [{"id": 0, "grid_id": grid["id"],
                       "data_source_ids": source_ids, "formats": [netcdf_id],
                       "left": 27.0, "right": 30.4, "bottom": 59.4, "top": 60.6}],
        }
        result = request_json("/api/queue", payload)
        manifest = {"grid": grid, "request": payload, "response": result,
                    "source": BASE, "retrieved_utc": datetime.now(timezone.utc).isoformat()}
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    basket = manifest["response"]["basketId"]
    archive = RAW / "gebco_2026_gulf.zip"
    if not archive.exists():
        for _ in range(60):
            status = request_json("/api/queue/status/" + basket)
            print(status, flush=True)
            if status["status"] == "finished":
                break
            if status["status"] not in {"new", "processing", "queued", "running"}:
                raise RuntimeError(f"Unexpected server status: {status}")
            time.sleep(5)
        else:
            raise RuntimeError("Subset not ready; rerun to resume the same request")
        temp = archive.with_suffix(".part")
        with urllib.request.urlopen(BASE + "/api/queue/download/" + basket, timeout=120) as r:
            with temp.open("wb") as out:
                while block := r.read(1024 * 1024):
                    out.write(block)
        with zipfile.ZipFile(temp) as z:
            if z.testzip() is not None:
                raise RuntimeError("Invalid download")
        temp.replace(archive)
    with zipfile.ZipFile(archive) as z:
        for item in z.infolist():
            if item.filename.lower().endswith(".nc"):
                target = RAW / Path(item.filename).name
                target.write_bytes(z.read(item))
                print(f"Extracted {target.name}: {target.stat().st_size} bytes")
    manifest["archive_sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
    manifest["archive_bytes"] = archive.stat().st_size
    manifest["archive_members"] = zipfile.ZipFile(archive).namelist()
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Saved download provenance:", manifest_path)


if __name__ == "__main__":
    main()
