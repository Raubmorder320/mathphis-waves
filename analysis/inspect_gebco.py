"""Stage 2: inspect the unmodified GEBCO subset before accepting it as input."""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset

ROOT = Path(__file__).resolve().parents[1]


def main():
    raw = ROOT / "data/raw"
    source = raw / "gebco_2026_n60.6_s59.4_w27.0_e30.4.nc"
    tid_path = raw / "gebco_2026_tid_n60.6_s59.4_w27.0_e30.4.nc"
    with Dataset(source) as ds:
        lon, lat = np.array(ds["lon"][:]), np.array(ds["lat"][:])
        assert ds["elevation"].dimensions == ("lat", "lon")
        z = np.array(ds["elevation"][:])
        attrs = {key: ds.getncattr(key) for key in ds.ncattrs()}
    with Dataset(tid_path) as td:
        assert np.array_equal(td["lon"][:], lon) and np.array_equal(td["lat"][:], lat)
        assert td["tid"].dimensions == ("lat", "lon")
        tid = np.array(td["tid"][:])
    ix, iy = np.argsort(lon), np.argsort(lat)
    lon, lat = lon[ix], lat[iy]
    z, tid = z[np.ix_(iy, ix)], tid[np.ix_(iy, ix)]
    assert z.shape == (len(lat), len(lon))
    assert np.isfinite(z).all()
    xx, yy = np.meshgrid(lon, lat)
    # Diagnostic rectangle in the interior of Neva Bay, not a definitive shoreline mask.
    bay = (xx >= 29.95) & (xx <= 30.15) & (yy >= 59.92) & (yy <= 60.02)
    vals, counts = np.unique(tid[bay], return_counts=True)
    samples = []
    for name, x, y in [("west_gulf", 27.5, 59.8),
                       ("west_of_kotlin", 29.5, 60.0),
                       ("bay_west", 29.98, 59.97),
                       ("bay_centre", 30.05, 59.97),
                       ("bay_east", 30.15, 59.95)]:
        ix, iy = np.argmin(abs(lon - x)), np.argmin(abs(lat - y))
        samples.append({"name": name, "lon": float(lon[ix]), "lat": float(lat[iy]),
                        "elevation_m": int(z[iy, ix]), "tid": int(tid[iy, ix])})
    report = {
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "tid_sha256": hashlib.sha256(tid_path.read_bytes()).hexdigest(),
        "shape_yx": list(z.shape),
        "lon_range": [float(lon[0]), float(lon[-1])],
        "lat_range": [float(lat[0]), float(lat[-1])],
        "spacing_degrees": [float(np.diff(lon).mean()), float(np.diff(lat).mean())],
        "elevation_range_m": [int(z.min()), int(z.max())],
        "bay_rectangle": [29.95, 30.15, 59.92, 60.02],
        "bay_elevation_percentiles_m": np.percentile(z[bay], [0, 25, 50, 75, 100]).tolist(),
        "bay_fraction_negative": float(np.mean(z[bay] < 0)),
        "bay_fraction_depth_3_to_5_m": float(np.mean((z[bay] >= -5) & (z[bay] <= -3))),
        "bay_tid_counts": {str(int(v)): int(n) for v, n in zip(vals, counts)},
        "samples": samples, "source_attributes": attrs,
    }
    figures = ROOT / "analysis/figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for ax, bounds, title in zip(axes, [(27, 30.4, 59.4, 60.6),
                                      (29.6, 30.3, 59.8, 60.15)],
                                 ["GEBCO 2026: unmodified source", "Neva Bay: unmodified source"]):
        im = ax.pcolormesh(lon, lat, np.ma.masked_where(z >= 0, -z),
                           cmap="Blues", vmin=0, vmax=50, shading="auto")
        ax.contourf(lon, lat, z >= 0, levels=[0.5, 1.5], colors=["#c7c3b8"])
        ax.set(xlim=bounds[:2], ylim=bounds[2:], xlabel="Longitude (degrees E)",
               ylabel="Latitude (degrees N)", title=title)
        ax.set_aspect(2)
    for sample in samples[2:]:
        axes[1].plot(sample["lon"], sample["lat"], "ro", ms=3)
        axes[1].annotate(f'{sample["elevation_m"]} m', (sample["lon"], sample["lat"]),
                         xytext=(3, 5), textcoords="offset points", fontsize=8)
    fig.colorbar(im, ax=axes, label="Depth (m); grey = source elevation >= 0")
    fig.savefig(figures / "gebco_source_check.png", dpi=160)
    plt.close(fig)
    (ROOT / "analysis/source_check.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "source_attributes"}, indent=2))


if __name__ == "__main__":
    main()
