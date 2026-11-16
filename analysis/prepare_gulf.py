"""Build the model-ready bathymetry and a section from the same reduced grid.

Run: python -m analysis.prepare_gulf. Inputs are the archived GEBCO subset.
No network access and no artificial depth clipping or dam correction.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import label

ROOT = Path(__file__).resolve().parents[1]
R = 6371008.8
WEST, EAST, SOUTH, NORTH = 27.0, 30.4, 59.4, 60.6
LAT0 = 60.0
NX, NY = 150, 50


def metres(lon, lat):
    """Local equirectangular coordinates, origin at the SW corner; metres."""
    return (R * np.cos(np.deg2rad(LAT0)) * np.deg2rad(np.asarray(lon) - WEST),
            R * np.deg2rad(np.asarray(lat) - SOUTH))


def build():
    source = ROOT / "data/raw/gebco_2026_n60.6_s59.4_w27.0_e30.4.nc"
    with Dataset(source) as ds:
        lon, lat = np.array(ds["lon"][:]), np.array(ds["lat"][:])
        assert ds["elevation"].dimensions == ("lat", "lon")
        z = np.array(ds["elevation"][:], dtype=float)
    assert np.isfinite(z).all()
    assert np.all(np.diff(lon) > 0) and np.all(np.diff(lat) > 0)
    lon_edges = np.linspace(WEST, EAST, NX + 1)
    lat_edges = np.linspace(SOUTH, NORTH, NY + 1)
    lon_out = (lon_edges[1:] + lon_edges[:-1]) / 2
    lat_out = (lat_edges[1:] + lat_edges[:-1]) / 2
    xx, yy = np.meshgrid(lon, lat)
    area = np.cos(np.deg2rad(yy))
    wet = z < 0

    def bin_sum(weights):
        return np.histogram2d(yy.ravel(), xx.ravel(), bins=(lat_edges, lon_edges),
                              weights=weights.ravel())[0]

    total_area = bin_sum(area)
    wet_area = bin_sum(area * wet)
    assert np.all(total_area > 0)
    fraction = wet_area / total_area
    depth_mean = np.divide(bin_sum(area * np.maximum(-z, 0)), wet_area,
                           out=np.zeros_like(wet_area), where=wet_area > 0)
    # Majority-water classification preserves land; mean over wet samples avoids
    # diluting actual depth by treating land as a water column of zero depth.
    h = np.where(fraction >= 0.5, depth_mean, 0)
    x, _ = metres(lon_out, SOUTH)
    _, y = metres(WEST, lat_out)
    components, _ = label(h > 0)  # four-neighbour connectivity, as for FV fluxes

    def cell(a, b):
        return np.argmin(abs(lat_out - b)), np.argmin(abs(lon_out - a))

    west_cell = cell(27.05, 60.08)
    bay_cell = cell(30.05, 59.97)
    kotlin_cell = cell(29.75, 60.005)
    assert components[west_cell] != 0
    assert components[west_cell] == components[bay_cell], "Bay isolated from gulf"
    assert h[kotlin_cell] == 0, "Kotlin lost during coarsening"
    bay = ((lon_out[None, :] >= 29.95) & (lon_out[None, :] <= 30.15)
           & (lat_out[:, None] >= 59.92) & (lat_out[:, None] <= 59.98) & (h > 0))
    assert bay.sum() > 0
    assert 3 <= np.median(h[bay]) <= 5, "Bay depth does not match project criterion"

    # Route follows open water and passes SOUTH of Kotlin, then enters Neva Bay.
    waypoints = np.array([(lon_out[0], 60.08), (28.0, 60.10), (28.8, 60.07),
                          (29.4, 60.05), (29.6, 59.97), (29.9, 59.96),
                          (30.15, 59.95), (30.30, 59.95)])
    wx, wy = metres(waypoints[:, 0], waypoints[:, 1])
    ws = np.r_[0., np.cumsum(np.hypot(np.diff(wx), np.diff(wy)))]
    s_probe = np.linspace(0, ws[-1], 20001)
    px = np.interp(s_probe, ws, wx)
    py = np.interp(s_probe, ws, wy)
    nearest = RegularGridInterpolator((y, x), h, method="nearest", bounds_error=True)
    probe = nearest(np.column_stack((py, px)))
    raw_nearest = RegularGridInterpolator((lat, lon), z, method="nearest", bounds_error=True)
    probe_lon = WEST + np.rad2deg(px / (R * np.cos(np.deg2rad(LAT0))))
    probe_lat = SOUTH + np.rad2deg(py / R)
    raw_probe = raw_nearest(np.column_stack((probe_lat, probe_lon)))
    # End at the first shoreline in either grid, avoiding a route that appears
    # wet after averaging but has already crossed dry land in the source.
    first_dry = np.flatnonzero((probe == 0) | (raw_probe >= 0))
    if len(first_dry) == 0:
        raise ValueError("Route has not reached a dry coastal cell")
    end = first_dry[0]
    end_lon = WEST + np.rad2deg(px[end] / (R * np.cos(np.deg2rad(LAT0))))
    if end_lon < 30.15:
        raise ValueError(f"Route intersects land before the intended shore: lon={end_lon}")
    # Boundary between the last wet and first dry nearest-pixel classifications.
    shore_s = (s_probe[end - 1] + s_probe[end]) / 2
    s = np.linspace(0., shore_s, 801)
    profile_x, profile_y = np.interp(s, ws, wx), np.interp(s, ws, wy)
    interp = RegularGridInterpolator((y, x), h, bounds_error=True)
    profile_h = interp(np.column_stack((profile_y, profile_x)))
    assert np.all(nearest(np.column_stack((profile_y[:-1], profile_x[:-1]))) > 0)
    assert np.all(profile_h[:-1] > 0)
    # Explicit coastal endpoint; this is not a surveyed beach slope.
    profile_h[-1] = 0.
    profile_lon = WEST + np.rad2deg(profile_x / (R * np.cos(np.deg2rad(LAT0))))
    profile_lat = SOUTH + np.rad2deg(profile_y / R)
    raw_profile = raw_nearest(np.column_stack((profile_lat, profile_lon)))
    assert np.all(raw_profile[:-1] < 0), "Section crosses source land"
    common = dict(source="GEBCO_2026", scenario="educational_no_barrier",
                  correction_applied=False, origin_lon=WEST, origin_lat=SOUTH,
                  reference_lat=LAT0, earth_radius_m=R)
    data = ROOT / "data"
    np.savez_compressed(data / "gulf_bathymetry.npz", x=x, y=y, h=h,
                        lon=lon_out, lat=lat_out, wet_fraction=fraction, **common)
    np.savez_compressed(data / "gulf_profile.npz", x=s, h=profile_h,
                        lon=profile_lon, lat=profile_lat, map_x=profile_x,
                        map_y=profile_y, waypoints_lonlat=waypoints, **common)
    report = {
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "shape_yx": list(h.shape), "dx_m": float(np.diff(x).mean()),
        "dy_m": float(np.diff(y).mean()),
        "bay_wet_cell_count": int(bay.sum()),
        "bay_depth_percentiles_m": np.percentile(h[bay], [0, 25, 50, 75, 100]).tolist(),
        "gulf_connected_to_bay": True, "kotlin_retained": True,
        "dam_correction_applied": False, "profile_nodes": len(s),
        "profile_length_m": float(s[-1]), "profile_dx_m": float(s[1]),
        "profile_shore_lonlat": [float(profile_lon[-1]), float(profile_lat[-1])],
        "profile_min_wet_depth_m": float(profile_h[:-1].min()),
        "profile_max_depth_m": float(profile_h.max()),
        "profile_start_depth_m": float(profile_h[0]),
        "nearshore_slope_10_to_1_km": float(
            (np.interp(s[-1] - 10000, s, profile_h)
             - np.interp(s[-1] - 1000, s, profile_h)) / 9000),
        "nearshore_slope_12_to_2_km": float(
            (np.interp(s[-1] - 12000, s, profile_h)
             - np.interp(s[-1] - 2000, s, profile_h)) / 10000),
        "profile_crosses_source_land": False,
        "shore_method": "first wet/dry boundary in either source or reduced grid",
        "waypoints_lonlat": waypoints.tolist(),
        "coarsening": "area-weighted wet mean; water fraction >= 0.5",
        "projection": "local equirectangular, R=6371008.8 m, lat0=60 deg",
    }
    for name in ["gulf_bathymetry.npz", "gulf_profile.npz"]:
        with np.load(data / name, allow_pickle=False) as result:
            assert np.isfinite(result["h"]).all() and np.all(result["h"] >= 0)
            assert np.all(np.diff(result["x"]) > 0)
        report[name] = {"bytes": (data / name).stat().st_size}
    (ROOT / "analysis/preparation_checks.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    figures = ROOT / "analysis/figures"
    figures.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for ax, zoom in zip(axes, [False, True]):
        im = ax.pcolormesh(x / 1000, y / 1000, np.ma.masked_where(h == 0, h),
                           cmap="Blues", vmin=0, vmax=50, shading="auto")
        ax.contourf(x / 1000, y / 1000, h == 0, levels=[.5, 1.5], colors=["#c7c3b8"])
        ax.plot(profile_x / 1000, profile_y / 1000, "r-", lw=1.2, label="1D section")
        ax.plot(profile_x[-1] / 1000, profile_y[-1] / 1000, "ro", ms=4)
        ax.set(xlabel="East from 27 E (km)", ylabel="North from 59.4 N (km)",
               title="Neva Bay: reduced grid" if zoom else "GEBCO 2026: grid 150 x 50")
        ax.set_aspect("equal")
        if zoom:
            a, b = metres(29.5, 59.8)
            c, d = metres(30.35, 60.16)
            ax.set(xlim=(a / 1000, c / 1000), ylim=(b / 1000, d / 1000))
        ax.legend(loc="lower left")
    fig.colorbar(im, ax=axes, label="Depth (m)")
    fig.savefig(figures / "gulf_bathymetry.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 4), constrained_layout=True)
    ax.fill_between(s / 1000, -profile_h, -95, color="#c7c3b8")
    ax.plot(s / 1000, -profile_h, color="navy")
    ax.axhline(0, color="steelblue", lw=.8)
    ax.set(xlabel="Distance along section from west (km)", ylabel="Bed elevation (m)",
           title="Section south of Kotlin to Neva Bay shore (from reduced grid)",
           xlim=(0, s[-1] / 1000), ylim=(-95, 2))
    ax.grid(alpha=.25)
    fig.savefig(figures / "gulf_profile.png", dpi=160)
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    build()
