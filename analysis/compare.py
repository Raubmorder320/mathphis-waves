"""Compare registered 1D models directly: python -m analysis.compare.

Figures go to analysis/figures, numerical results to analysis/results.
The default experiment implements the Sunday task: linear vs Saint-Venant.
Neither model implementations nor their shared initial conditions are changed.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from backend.common import G, N_PACKET, load_gulf_profile, setup, u_linear, u_simple_wave
from backend.contract import Params
from backend.registry import MODELS

ROOT = Path(__file__).resolve().parents[1]
THRESHOLD = 0.1


@dataclass
class Comparison:
    x: np.ndarray
    t: np.ndarray
    h: np.ndarray
    difference: np.ndarray  # |eta_model - eta_reference| / input amplitude
    d: np.ndarray          # max over ALL spatial nodes, as required by TASKS.md
    coastal_response: float | None
    initial_velocity_difference: float


def first_exceedance(t: np.ndarray, values: np.ndarray, threshold=THRESHOLD):
    """First SAVED time with a strict > threshold; None means right-censored.

    Also return the last preceding saved time, so temporal uncertainty is explicit.
    No interpolation or assumption that a missing crossing occurs at t_end.
    """
    t, values = np.asarray(t), np.asarray(values)
    if t.ndim != 1 or values.shape != t.shape or not len(t):
        raise ValueError("Expected matching nonempty 1D time and value arrays")
    if not np.isfinite(values).all() or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError("Nonfinite values or non-increasing times")
    indices = np.flatnonzero(values > threshold)
    if not len(indices):
        return None, None
    j = int(indices[0])
    return float(t[j]), float(t[j - 1]) if j else float(t[0])


def solve_pair(p: Params, model="saint_venant", reference="linear") -> Comparison:
    """Use the public registry/Params/Result interface, without a web server."""
    if p.a <= 0:
        raise ValueError("A positive amplitude is required for normalization")
    arrays = []
    for model_id in (reference, model):
        if model_id not in MODELS or MODELS[model_id].dim != 1:
            raise ValueError(f"Unavailable 1D model: {model_id}")
        result = MODELS[model_id].solve(replace(p, extra=dict(p.extra)))
        arrays.append(tuple(np.asarray(v, dtype=float) for v in
                            (result.x, result.t, result.h, result.eta)))
        del result  # do not retain large Python lists alongside the arrays
    x, t, h, eta_ref = arrays[0]
    x2, t2, h2, eta_model = arrays[1]
    if not all(np.array_equal(a, b) for a, b in ((x, x2), (t, t2), (h, h2))):
        raise ValueError("Models returned different coordinates, times or bathymetry")
    if eta_ref.shape != (len(t), len(x)) or eta_model.shape != eta_ref.shape:
        raise ValueError("Invalid eta array shape")
    if not all(np.isfinite(a).all() for a in (x, t, h, eta_ref, eta_model)):
        raise ValueError("Nonfinite model result")
    if not np.allclose(eta_ref[0], eta_model[0], atol=p.a * 1e-9, rtol=0):
        raise ValueError("Models returned different initial surface elevations")
    difference = np.abs(eta_model - eta_ref) / p.a
    d = difference.max(axis=1)
    coast = None
    if p.bottom != "flat":
        wet = np.flatnonzero(h > 0)
        # Once the last initially wet node responds, boundary conditions differ:
        # linear reflects there, Saint-Venant permits shoreline movement.
        signal = np.maximum(abs(eta_ref[:, wet[-1]]), abs(eta_model[:, wet[-1]])) / p.a
        coast, _ = first_exceedance(t, signal, threshold=1e-3)
    s = setup(p)
    du = float(np.max(abs(u_linear(s.h, s.eta0) - u_simple_wave(s.h, s.eta0))))
    return Comparison(x, t, h, difference, d, coast, du)


def crossing_record(c: Comparison, a: float, coastal_censor=False):
    t10, lower = first_exceedance(c.t, c.d)
    record = {"amplitude_m": float(a), "t10_s": t10, "t10_previous_frame_s": lower,
              "coastal_response_s": c.coastal_response,
              "initial_velocity_difference_m_s": c.initial_velocity_difference}
    valid = t10 is not None and (not coastal_censor or c.coastal_response is None
                                or t10 < c.coastal_response)
    record["status"] = ("crossed" if valid else "not_reached" if t10 is None
                        else "coastal_boundary_already_active")
    if valid:
        j = int(np.searchsorted(c.t, t10))
        ix = int(np.argmax(c.difference[j]))
        record.update(onset_x_m=float(c.x[ix]), onset_depth_m=float(c.h[ix]),
                      distance_to_shore_m=float(c.x[-1] - c.x[ix]))
    else:
        record.update(onset_x_m=None, onset_depth_m=None, distance_to_shore_m=None)
    return record


def breaking_time(a, p):
    return p.wavelength / (3 * np.pi * (np.asarray(a) / p.depth) * np.sqrt(G * p.depth))


def run_study(amplitudes, params, model, reference):
    records, all_d, spatial_times, spatial_peaks = [], [], [], []
    for a in amplitudes:
        start = time.perf_counter()
        c = solve_pair(replace(params, a=float(a)), model, reference)
        record = crossing_record(c, a, coastal_censor=params.bottom == "gulf")
        if params.bottom == "flat":
            # A periodic domain has no shoreline, and identical periods have
            # equivalent spatial maxima. Only temporal onset is meaningful here.
            record.update(onset_x_m=None, onset_depth_m=None, distance_to_shore_m=None)
        record["elapsed_s"] = round(time.perf_counter() - start, 3)
        records.append(record)
        all_d.append(c.d)
        valid_t = np.ones(len(c.t), dtype=bool)
        if params.bottom == "gulf" and c.coastal_response is not None:
            valid_t &= c.t < c.coastal_response
        exceed = c.difference[valid_t] > THRESHOLD
        valid_times = c.t[valid_t]
        first = np.where(exceed.any(axis=0), valid_times[exceed.argmax(axis=0)], np.nan)
        spatial_times.append(first)
        spatial_peaks.append(c.difference[valid_t].max(axis=0))
        print(f"{params.bottom}: a={a:.4g} m, t10={record['t10_s']}, "
              f"status={record['status']}, {record['elapsed_s']:.2f} s", flush=True)
    return c.x, c.t, c.h, np.asarray(all_d), np.asarray(spatial_times), np.asarray(spatial_peaks), records


def refinement_checks(amplitudes, params, records, model, reference):
    """Check end amplitudes with dx and output interval halved simultaneously."""
    checks = []
    for i in (0, len(amplitudes) - 1):
        fine = replace(params, a=float(amplitudes[i]),
                       nx=(2 * params.nx if params.bottom == "flat" else 2 * params.nx - 1),
                       n_frames=2 * params.n_frames - 1)
        # Only need the interval through onset for the refinement calculation.
        onset = records[i]["t10_s"]
        if onset is not None:
            dt = params.t_end / (params.n_frames - 1) / 2
            fine.t_end = min(params.t_end, max(1.5 * onset, onset + 4 * dt))
            fine.n_frames = int(np.ceil(fine.t_end / dt)) + 1
        c = solve_pair(fine, model, reference)
        r = crossing_record(c, amplitudes[i], coastal_censor=params.bottom == "gulf")
        change = None if onset is None or r["t10_s"] is None else abs(r["t10_s"] / onset - 1)
        xchange = None if r["onset_x_m"] is None or records[i]["onset_x_m"] is None else abs(
            r["onset_x_m"] - records[i]["onset_x_m"])
        checks.append({"amplitude_m": float(amplitudes[i]), "fine_nx": fine.nx,
                       "fine_frame_interval_s": float(c.t[1] - c.t[0]),
                       "fine_t10_s": r["t10_s"], "relative_t10_change": change,
                       "onset_x_change_m": xchange})
        print(f"refinement {params.bottom}: a={amplitudes[i]:g}, "
              f"relative t10 change={change}", flush=True)
    return checks


def plot_results(flat, gulf, flat_a, gulf_a, params, figures,
                 model="saint_venant", reference="linear"):
    fx, ft, fh, fd, _, _, fr = flat
    gx, gt, gh, gd, _, _, gr = gulf
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    im = ax.pcolormesh(ft, flat_a, fd, shading="nearest", cmap="magma", vmin=0, vmax=1)
    ax.contour(ft, flat_a, fd, levels=[THRESHOLD], colors="cyan", linewidths=1.2)
    smooth_a = np.linspace(flat_a[0], flat_a[-1], 500)
    ax.plot(breaking_time(smooth_a, params), smooth_a, "w--", label="Теория: t*(a)")
    ax.plot([r["t10_s"] or np.nan for r in fr], flat_a, "co", ms=3,
            label="Первый кадр d > 0,1")
    ax.set(xlim=(0, ft[-1]), ylim=(flat_a[0], flat_a[-1]),
           xlabel="Время, с", ylabel="Начальная амплитуда, м",
           title=f"{MODELS[model].name} и {MODELS[reference].name}: ровное дно")
    ax.legend(loc="upper right", facecolor="0.2", labelcolor="white", framealpha=.9)
    fig.colorbar(im, ax=ax, label="d(t) = max |ηмодель − ηбаза| / a (цвет ограничен 1)")
    fig.savefig(figures / "compare_flat_map.png", dpi=160)
    plt.close(fig)

    t10 = np.array([r["t10_s"] if r["t10_s"] is not None else np.nan for r in fr])
    valid = np.isfinite(t10) & (t10 > 0)
    slope = intercept = None
    if valid.sum() >= 2:
        slope, intercept = np.polyfit(np.log(flat_a[valid]), np.log(t10[valid]), 1)
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.loglog(flat_a[valid], t10[valid], "o-", label="Расчёт: первый кадр d > 0,1")
    if valid.any():
        inverse = np.exp(np.mean(np.log(t10[valid] * flat_a[valid]))) / flat_a
        ax.loglog(flat_a, inverse, "--", label="Зависимость 1/a, наклон −1")
    if slope is not None:
        ax.loglog(flat_a, np.exp(intercept) * flat_a ** slope, ":",
                  label=f"Аппроксимация: наклон {slope:.3f}")
    ax.set(xlabel="Начальная амплитуда, м", ylabel="t₁₀, с",
           title="Время достижения расхождения 10%")
    ax.grid(True, which="both", alpha=.25)
    ax.legend()
    fig.savefig(figures / "compare_t10.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5.5), constrained_layout=True)
    distance = (gx[-1] - gx) / 1000
    ax.plot(distance, -gh, color="0.25", label="Дно по профилю GEBCO")
    ax.fill_between(distance, -gh, -max(gh) * 1.1, color="0.85")
    for r, color in zip(gr, plt.cm.viridis(np.linspace(.05, .95, len(gr)))):
        if r["status"] == "crossed":
            ax.plot(r["distance_to_shore_m"] / 1000, -r["onset_depth_m"], "o",
                    color=color, ms=6,
                    label=f"a={r['amplitude_m']:.2f} м; t={r['t10_s']:.0f} с")
        else:
            ax.plot([], [], "x", color=color, label=f"a={r['amplitude_m']:.2f} м: нет границы")
    ax.set(xlim=(distance[0], 0), ylim=(-max(gh) * 1.1, 2),
           xlabel="Расстояние до берега вдоль разреза, км", ylabel="Отметка дна, м",
           title="Залив: место первого достижения расхождения 10%")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    ax.grid(alpha=.2)
    fig.savefig(figures / "compare_gulf_boundary.png", dpi=160)
    plt.close(fig)
    return float(slope) if slope is not None else None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="saint_venant")
    parser.add_argument("--reference", default="linear")
    parser.add_argument("--flat-nx", type=int, default=3200)
    parser.add_argument("--gulf-nx", type=int, default=3201)
    parser.add_argument("--figures-dir", type=Path, default=ROOT / "analysis/figures")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "analysis/results")
    args = parser.parse_args(argv)
    if args.flat_nx < 40 or args.gulf_nx < 40:
        parser.error("At least 40 spatial nodes are required")
    if args.model == args.reference:
        parser.error("Choose two different models")
    for folder in (args.figures_dir, args.results_dir):
        folder.mkdir(parents=True, exist_ok=True)
    flat_a = np.geomspace(.05, 1.5, 10)
    gulf_a = np.linspace(.2, 2., 10)
    flat_p = Params(bottom="flat", wavelength=100, depth=5, L=1000,
                    nx=args.flat_nx, t_end=160, n_frames=1601)
    gulf_p = Params(bottom="gulf", wavelength=10000, nx=args.gulf_nx,
                    t_end=7000, n_frames=701)
    s = setup(gulf_p)
    if N_PACKET * gulf_p.wavelength >= s.x[-1] - s.x[0] or np.max(abs(s.eta0)) == 0:
        raise ValueError("The initial gulf packet does not fit or is unresolved")
    summaries, studies = {}, []
    for name, amplitudes, p in (("flat", flat_a, flat_p), ("gulf", gulf_a, gulf_p)):
        study = run_study(amplitudes, p, args.model, args.reference)
        studies.append(study)
        x, t, h, d, local_times, peaks, records = study
        checks = refinement_checks(amplitudes, p, records, args.model, args.reference)
        floor = solve_pair(replace(p, a=1e-4), args.model, args.reference)
        floor_max = float(floor.d.max())
        print(f"{name}: small-amplitude control max d={floor_max:.5f}", flush=True)
        np.savez_compressed(args.results_dir / f"compare_{name}.npz",
                            amplitudes=amplitudes, x=x, t=t, h=h, d=d,
                            first_local_crossing_s=local_times,
                            peak_local_difference_before_coast=peaks,
                            small_amplitude_d=floor.d, threshold=THRESHOLD,
                            reference=args.reference, model=args.model,
                            t_star=breaking_time(amplitudes, p) if name == "flat" else np.full(len(amplitudes), np.nan))
        summaries[name] = {"params": asdict(p), "records": records,
                           "refinement": checks, "small_amplitude_control_a_m": 1e-4,
                           "small_amplitude_control_max_d": floor_max,
                           "actual_domain_length_m": float(p.L if name == "flat" else x[-1] - x[0]),
                           "initial_surface_peak_over_a": float(np.max(abs(setup(p).eta0)) / p.a),
                           "max_kh": float(2 * np.pi * h.max() / p.wavelength)}
    warnings = []
    for name in ("flat", "gulf"):
        if summaries[name]["max_kh"] > .3:
            warnings.append(f"{name}: kh exceeds 0.3; marginal shallow-water benchmark")
        if summaries[name]["small_amplitude_control_max_d"] >= .03:
            warnings.append(f"{name}: small-amplitude control exceeds 3%; refine the grid")
        for check in summaries[name]["refinement"]:
            if check["relative_t10_change"] is None or check["relative_t10_change"] > .05:
                warnings.append(f"{name}: t10 refinement is unavailable or changes more than 5%")
    slope = plot_results(*studies, flat_a, gulf_a, flat_p, args.figures_dir,
                         args.model, args.reference)
    summary = {"schema_version": 1, "reference": args.reference, "model": args.model,
               "threshold": THRESHOLD, "flat_loglog_slope": slope, **summaries,
               "warnings": warnings,
               "gulf_profile_sha256": hashlib.sha256((ROOT / "data/gulf_profile.npz").read_bytes()).hexdigest(),
               "onset_definition": "first saved global d>0.1; location is argmax at that frame",
               "gulf_coastal_censor": "onset must precede |eta|/a>1e-3 at last initially wet node",
               "initial_velocities": "model-native u_linear and u_simple_wave from common.py"}
    (args.results_dir / "compare_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for warning in warnings:
        print("WARNING: " + warning, flush=True)
    print(f"Saved three figures to {args.figures_dir}; flat log-log slope={slope}", flush=True)


if __name__ == "__main__":
    main()
