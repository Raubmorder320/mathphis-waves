"""Third-wave research: python -m analysis.third_wave.

Uses existing solvers; does not modify model equations, shared setup or contracts.
Numerical experiments are separate from cited historical observations.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import re
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from backend.common import G, N_PACKET, setup
from backend.contract import Params
from backend.registry import MODELS
from backend.models.shallow_2d import SPB, gauge, load_gulf, solve_2d
from analysis.compare import first_exceedance

ROOT = Path(__file__).resolve().parents[1]
MODEL_IDS = ("saint_venant", "kdv", "boussinesq")
COLORS = {"linear": "#555555", "saint_venant": "#23954f", "kdv": "#cf6421",
          "boussinesq": "#276bb5", "boussinesq_no_breaking": "#ad4d9c"}
NAMES = {"linear": "Линейная", "saint_venant": "Сен-Венан", "kdv": "КдФ",
         "boussinesq": "Буссинеск", "boussinesq_no_breaking": "Буссинеск, γ=0"}


def write_json(path, data):
    """JSON null represents unavailable data, never a fabricated value."""
    def clean(x):
        if isinstance(x, dict):
            return {k: clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        if isinstance(x, (float, np.floating)):
            return float(x) if np.isfinite(x) else None
        if isinstance(x, np.integer):
            return int(x)
        return x
    path.write_text(json.dumps(clean(data), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def field(p, model_id, gamma=None):
    extra = dict(p.extra)
    if gamma is not None:
        extra["gamma"] = gamma
    r = MODELS[model_id].solve(replace(p, extra=extra))
    eta = np.asarray(r.eta, float)
    s = setup(p)
    if (eta.shape != (p.n_frames, p.nx) or not np.isfinite(eta).all()
            or not np.array_equal(r.x, s.x) or not np.array_equal(r.h, s.h)
            or not np.array_equal(r.t, np.linspace(0., p.t_end, p.n_frames))
            or not np.allclose(eta[0], s.eta0, atol=p.a * 1e-9, rtol=0)):
        raise ValueError(f"Incompatible or invalid model output: {model_id}")
    return np.asarray(r.t), eta


def align_periodic(reference, target):
    """Rigid periodic translation only; no nonlinear stretching of profiles.

    Cross-correlation finds the translation, a parabolic fit gives subcell shift.
    Returns the translated target and the applied shift measured in grid cells.
    """
    reference, target = np.asarray(reference), np.asarray(target)
    if reference.shape != target.shape or reference.ndim != 1:
        raise ValueError("Expected equal one-dimensional periodic profiles")
    corr = np.fft.irfft(np.fft.rfft(reference) * np.conj(np.fft.rfft(target)), len(target))
    i = int(np.argmax(corr))
    left, mid, right = corr[(i - 1) % len(corr)], corr[i], corr[(i + 1) % len(corr)]
    den = left - 2 * mid + right
    fraction = 0.5 * (left - right) / den if abs(den) > np.finfo(float).eps * max(abs(mid), 1e-30) else 0.
    shift = (i + fraction + len(target) / 2) % len(target) - len(target) / 2
    moved = np.fft.irfft(np.fft.rfft(target) * np.exp(-2j * np.pi * np.fft.rfftfreq(len(target)) * shift), len(target))
    return moved, float(shift)


def pair_metrics(reference, other, a):
    moved, shift = align_periodic(reference, other)
    return {"raw_linf_over_a": float(np.max(abs(other - reference)) / a),
            "aligned_rms_over_a": float(np.sqrt(np.mean((moved - reference) ** 2)) / a),
            "alignment_shift_cells": shift}, moved


def dispersion_case(ur, nx):
    # Fixed epsilon; lambda varies with Ur. Even Ur=1 has kh<0.3.
    h, a = 5., .0075
    lam = float(np.sqrt(ur * h ** 3 / a))
    tstar = lam / (3 * np.pi * (a / h) * np.sqrt(G * h))
    p = Params(bottom="flat", a=a, depth=h, wavelength=lam, L=4 * lam,
               nx=nx, t_end=2 * tstar, n_frames=121, extra={"gamma": 0.})
    values = {}
    for model in MODEL_IDS:
        t, eta = field(p, model)
        values[model] = eta
    reference = values["saint_venant"]
    comparisons, aligned = {}, {"saint_venant": reference[-1]}
    for model in ("kdv", "boussinesq"):
        comparisons[model], aligned[model] = pair_metrics(reference[-1], values[model][-1], a)
        comparisons[model]["max_d_over_time"] = float(abs(values[model] - reference).max() / a)
    return p, t, values, comparisons, aligned


def study_dispersion(out, figures, nx):
    urs = np.array([1., 4., 16., 64., 256.])
    records, storage = [], {"ursell": urs}
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for i, ur in enumerate(urs):
        start = time.perf_counter()
        p, t, values, metrics, aligned = dispersion_case(ur, nx)
        fine_metrics = None
        if i in (0, 2, 4):
            _, _, fine_values, fine_metrics, _ = dispersion_case(ur, nx * 2)
            for model in ("kdv", "boussinesq"):
                metrics[model]["refined_aligned_rms_over_a"] = fine_metrics[model]["aligned_rms_over_a"]
                metrics[model]["refinement_change"] = abs(fine_metrics[model]["aligned_rms_over_a"] - metrics[model]["aligned_rms_over_a"])
            del fine_values
        # gamma is inactive for this weak flat-wave experiment; verify on Ur=16.
        gamma_control = None
        if ur == 16:
            _, with_gamma = field(p, "boussinesq", gamma=.78)
            gamma_control = float(abs(with_gamma - values["boussinesq"]).max() / p.a)
        records.append({"ur": ur, "epsilon": p.a / p.depth,
                        "kh": 2 * np.pi * p.depth / p.wavelength,
                        "params": asdict(p), "t_star_s": p.t_end / 2,
                        "comparisons_to_saint_venant": metrics,
                        "gamma_control_max_difference_over_a": gamma_control})
        xx = np.arange(p.nx // 4) / (p.nx // 4)
        for model in MODEL_IDS:
            axes.flat[i].plot(xx, aligned[model][:p.nx // 4] / p.a, color=COLORS[model], label=NAMES[model])
            storage[f"case_{i}_{model}_final"] = values[model][-1]
            storage[f"case_{i}_{model}_d_from_sv"] = abs(values[model] - values["saint_venant"]).max(axis=1) / p.a
        storage[f"case_{i}_t"] = t
        axes.flat[i].set(title=f"Ur={ur:g}; kh={records[-1]['kh']:.3f}", xlabel="x / λ (один период)", ylabel="η / a")
        axes.flat[i].grid(alpha=.2)
        print(f"dispersion Ur={ur:g}: {time.perf_counter()-start:.1f} s", flush=True)
    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(handles, labels, loc="center")
    fig.suptitle("Форма волн при t=2t*: после совмещения с Сен-Венаном")
    fig.savefig(figures / "third_dispersion_profiles.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for model in ("kdv", "boussinesq"):
        for ax, key in zip(axes, ("raw_linf_over_a", "aligned_rms_over_a")):
            ax.semilogx(urs, [r["comparisons_to_saint_venant"][model][key] for r in records],
                        "o-", color=COLORS[model], label=NAMES[model])
    axes[0].set(title="Разность с учётом фазового сдвига", ylabel="max |ηмодель − ηСВ| / a")
    axes[1].set(title="Разность формы после совмещения", ylabel="RMS разности / a")
    for ax in axes:
        ax.set_xlabel("Число Урселла")
        ax.grid(alpha=.25)
        ax.legend()
    fig.savefig(figures / "third_dispersion_metrics.png", dpi=160)
    plt.close(fig)
    np.savez_compressed(out / "third_dispersion.npz", **storage)
    report = {"records": records, "phase_alignment": "rigid periodic FFT translation; diagnostic only",
              "breaking": "gamma=0; epsilon fixed at 0.0015", "snapshot": "t=2*t_star"}
    write_json(out / "third_dispersion.json", report)
    return report


def incoming_metrics(x, h, t, fields, a, wavelength, stop_s=None):
    """Conservative common window: strictly before the first coastal response.

    Cutoff respects each model's actual wet-domain wall. Surface maxima and
    front slopes refer only to this common window, not full storm envelopes.
    """
    coastal = {}
    for model, eta in fields.items():
        cutoff = max(a, .25) if model.startswith("boussinesq") else 0.
        inactive = np.flatnonzero(h < cutoff) if cutoff else np.flatnonzero(h <= 0)
        last = int(inactive[0] - 1) if len(inactive) else len(h) - 1
        hits = np.flatnonzero(abs(eta[:, last]) > .001 * a)
        coastal[model] = float(t[hits[0]]) if len(hits) else None
    hits = [v for v in coastal.values() if v is not None]
    stop = min(hits) if hits else float(t[-1]) + 1
    if stop_s is not None:
        stop = min(stop, stop_s)
    valid = t < stop
    if valid.sum() < 2:
        raise ValueError("No pre-boundary observation window")
    # Ignore weak unvisited tails when dividing slopes. Report only downstream
    # of the initially occupied packet, to avoid classifying its initial shape.
    visited = abs(fields["linear"][valid]).max(axis=0) >= .1 * a
    admissible = visited & (h >= max(a, .25)) & (x > x[0] + N_PACKET * wavelength)
    peaks, slopes, ratios = {}, {}, {}
    for model, eta in fields.items():
        peaks[model] = np.where(visited, np.maximum(eta[valid].max(axis=0), 0), np.nan)
        slopes[model] = np.maximum(-np.gradient(eta[valid], x, axis=1, edge_order=2), 0).max(axis=0)
    for model in fields:
        ratios[model] = np.divide(slopes[model], slopes["linear"], out=np.full_like(h, np.nan),
                                  where=admissible & (slopes["linear"] > 1e-12))
    crossings = np.flatnonzero(admissible & (ratios["saint_venant"] > 2))
    boundary = None
    if len(crossings):
        ix = int(crossings[0])
        previous = np.flatnonzero(admissible & (np.arange(len(x)) < ix))
        boundary = {"x_m": float(x[ix]), "distance_to_shore_m": float(x[-1] - x[ix]),
                    "depth_m": float(h[ix]), "ratio": float(ratios["saint_venant"][ix]),
                    "left_censored_at_packet_edge": not len(previous)}
    return peaks, ratios, {"coastal_response_s": coastal, "common_cutoff_s": float(stop),
                           "last_included_frame_s": float(t[valid][-1]),
                           "front_doubling": boundary,
                           "definition": "first downstream visited node with max incoming negative slope / linear >2"}


def gulf_case(a, nx, frames, include_control=True, return_fields=False):
    p = Params(bottom="gulf", a=float(a), wavelength=10000., nx=nx, t_end=18000., n_frames=frames)
    s = setup(p)
    fields = {}
    for model in ("linear", "saint_venant", "boussinesq"):
        t, fields[model] = field(p, model)
    if include_control:
        try:
            _, fields["boussinesq_no_breaking"] = field(p, "boussinesq", gamma=0.)
        except RuntimeError as error:
            # An unstable control must remain a failure, not be filled with zeros.
            control_error = str(error)
        else:
            control_error = None
    else:
        control_error = "not requested for refinement"
    peaks, ratios, record = incoming_metrics(s.x, s.h, t, fields, p.a, p.wavelength)
    record.update(amplitude_m=p.a, params=asdict(p), control_error=control_error)
    if "boussinesq_no_breaking" in fields:
        valid = t < record["common_cutoff_s"]
        record["breaking_control_max_difference_over_a"] = float(
            abs(fields["boussinesq"][valid] - fields["boussinesq_no_breaking"][valid]).max() / p.a)
    result = (s.x, s.h, peaks, ratios, record)
    return (*result, t, fields) if return_fields else result


def study_gulf(out, figures, nx):
    amplitudes = np.array([.2, .35, .5, .7, 1., 1.3, 1.6, 2.])
    records, storage = [], {"amplitudes": amplitudes}
    for i, a in enumerate(amplitudes):
        start = time.perf_counter()
        refine = a in (.5, 2.)
        case = gulf_case(a, nx, 721, return_fields=refine)
        x, h, peaks, ratios, record = case[:5]
        if a in (.5, 2.):
            xf, hf, pf, rf, fine, tf, fieldsf = gulf_case(a, 2 * nx - 1, 1441,
                                                        include_control=(a == .5), return_fields=True)
            tc, fieldsc = case[5:]
            shared_stop = min(record['common_cutoff_s'], fine['common_cutoff_s'])
            pc, _, _ = incoming_metrics(x, h, tc, fieldsc, a, 10000., stop_s=shared_stop)
            pf_shared, _, _ = incoming_metrics(xf, hf, tf, fieldsf, a, 10000., stop_s=shared_stop)
            refinement = {}
            matched = {}
            for model in ("linear", "saint_venant", "boussinesq"):
                projected = np.interp(x, xf, pf[model])
                valid = np.isfinite(projected) & np.isfinite(peaks[model]) & (h >= max(a, .25))
                refinement[model] = float(abs(projected[valid] - peaks[model][valid]).max() / a) if valid.any() else None
                projected = np.interp(x, xf, pf_shared[model])
                valid = np.isfinite(projected) & np.isfinite(pc[model]) & (h >= max(a, .25))
                offshore = valid & (h >= 5.) & (x < x[-1] - 5000.)
                matched[model] = {
                    'all_visited_linf_over_a': float(abs(projected[valid] - pc[model][valid]).max() / a),
                    'depth_ge_5m_linf_over_a': float(abs(projected[offshore] - pc[model][offshore]).max() / a)}
                storage[f'case_{i}_{model}_fine_peak_shared'] = pf_shared[model]
                storage[f'case_{i}_{model}_coarse_peak_shared'] = pc[model]
            record["refinement_peak_linf_over_a"] = refinement
            record['refinement_shared_cutoff_s'] = shared_stop
            record['refinement_matched_window'] = matched
            record["refined_front_doubling"] = fine["front_doubling"]
            record["refinement_note"] = "each run uses its measured pre-boundary cutoff; includes window sensitivity"
            record['refined_cutoff_s'] = fine['common_cutoff_s']
            storage[f'case_{i}_fine_x'] = xf
            del case, fieldsc, fieldsf
        records.append(record)
        for model in peaks:
            storage[f"case_{i}_{model}_peak"] = peaks[model]
            storage[f"case_{i}_{model}_front_ratio"] = ratios[model]
        print(f"gulf a={a:g}: cutoff={record['common_cutoff_s']:.0f} s; "
              f"front={record['front_doubling']}; {time.perf_counter()-start:.1f} s", flush=True)
    storage.update(x=x, h=h)
    np.savez_compressed(out / "third_gulf.npz", **storage)
    report = {"records": records,
              "warning": "front ratio is additional to d(t), not its replacement; unresolved coastal dispersion is not validated",
              "window": "common interval strictly before first 0.001*a response at any model's actual wall"}
    write_json(out / "third_gulf.json", report)
    plot_gulf(out, figures)
    return report


def plot_gulf(out, figures):
    records = json.loads((out / 'third_gulf.json').read_text(encoding='utf-8'))['records']
    with np.load(out / 'third_gulf.npz', allow_pickle=False) as arrays:
        x, h = arrays['x'], arrays['h']
        representative = {m: arrays[f'case_2_{m}_peak'] for m in
                          ('linear', 'saint_venant', 'boussinesq', 'boussinesq_no_breaking')}
        fine_x = arrays.get('case_2_fine_x')
        fine_peak = arrays.get('case_2_boussinesq_fine_peak_shared')
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, constrained_layout=True)
    distance = (x[-1] - x) / 1000
    axes[0].plot(distance, -h, color="0.25")
    axes[0].fill_between(distance, -h, -h.max() * 1.1, color=".85")
    for record, color in zip(records, plt.cm.viridis(np.linspace(.05, .95, len(records)))):
        b = record["front_doubling"]
        if b:
            axes[0].plot(b["distance_to_shore_m"] / 1000, -b["depth_m"], "o", color=color,
                         label=f"a={record['amplitude_m']:g} м" + (" (граница окна)" if b['left_censored_at_packet_edge'] else ""))
    axes[0].set(ylabel="Отметка дна, м", title="Профиль залива и первое удвоение крутизны фронта")
    axes[0].legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    for model, peak in representative.items():
        axes[1].plot(distance, peak, "--" if model.endswith("no_breaking") else "-",
                     color=COLORS[model], label=NAMES[model])
    if fine_x is not None:
        axes[1].plot((fine_x[-1] - fine_x) / 1000, fine_peak, color='#102f69', ls='--', lw=1,
                     label=f'Буссинеск, {len(fine_x)} узлов')
    axes[1].set(xlim=(distance[0], 0), xlabel="Расстояние до берега, км", ylabel="Максимум по сохранённым кадрам, м",
                title="Огибающие при a=0,5 м: сходимость Буссинеска не установлена")
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    for ax in axes:
        ax.grid(alpha=.2)
    fig.savefig(figures / "third_gulf_envelopes.png", dpi=160)
    plt.close(fig)


def historical_floods(path):
    """Read the already sourced catalogue, preserving original datum wording."""
    text = path.read_text(encoding="utf-8")
    definitions = dict(re.findall(r"^\[([^\]]+)\]:\s+(\S+)", text, re.M))
    table = text.split("## 4.", 1)[1].split("## 5.", 1)[0]
    records = []
    for line in table.splitlines():
        if not line.startswith("|"):
            continue
        cells = [v.strip() for v in line.strip("|").split("|")]
        if len(cells) != 4 or not re.search(r"\b(?:17|18|19|20)\d{2}\b", cells[0]):
            continue
        value = re.search(r"(\d+,\d+)\s*м", cells[1])
        if not value:
            raise ValueError(f"Missing sourced level: {line}")
        source_ids = re.findall(r"\]\[([^\]]+)\]", cells[3])
        if not source_ids or any(k not in definitions for k in source_ids):
            raise ValueError(f"Missing source link: {line}")
        records.append({"event": cells[0], "reported_level_m": float(value[1].replace(",", ".")),
                        "datum_as_published": cells[1], "wind_as_published": cells[2],
                        "sources": [definitions[k] for k in source_ids]})
    if len(records) < 2:
        raise ValueError("Historical comparison needs multiple sourced events")
    return records


def study_gulf_sampling(out, nx):
    """Separate output-cadence error from spatial-grid and window changes."""
    records, storage = {}, {}
    p = Params(bottom='gulf', a=.5, wavelength=10000., nx=nx, t_end=18000., n_frames=721)
    s = setup(p)
    # Same physical window, well before the measured response at the coastal wall.
    stop = 12000.
    for model in ('linear', 'boussinesq'):
        peaks = []
        for frames in (721, 2881):
            t, eta = field(replace(p, n_frames=frames), model)
            cutoff = max(p.a, .25) if model == 'boussinesq' else 0.
            inactive = np.flatnonzero(s.h < cutoff) if cutoff else np.flatnonzero(s.h <= 0)
            last = int(inactive[0] - 1)
            if np.any(abs(eta[t < stop, last]) > .001 * p.a):
                raise ValueError('Sampling-control window reached the coastal boundary')
            peak = np.maximum(eta[t < stop].max(axis=0), 0.)
            peaks.append(peak)
            storage[f'{model}_frames_{frames}_peak'] = peak
            del eta
        valid = (s.x > 30000.) & (s.h >= 5.) & (peaks[0] >= .1 * p.a)
        records[model] = {
            'peak_linf_change_over_a': float(abs(peaks[1][valid] - peaks[0][valid]).max() / p.a),
            'maximum_coarse_peak_m': float(peaks[0][valid].max()),
            'maximum_dense_peak_m': float(peaks[1][valid].max())}
        print(f'gulf sampling {model}: {records[model]}', flush=True)
    storage.update(x=s.x, h=s.h)
    np.savez_compressed(out / 'third_gulf_sampling.npz', **storage)
    write_json(out / 'third_gulf_sampling.json', {
        'params': asdict(p), 'stop_exclusive_s': stop, 'output_intervals_s': [25., 6.25],
        'region': 'x>30 km, h>=5 m, coarse positive peak>=0.1*a', 'records': records,
        'note': 'Changing frame times also changes the subdivision of internal solver steps'})


def study_surge(out, figures):
    observations = historical_floods(ROOT / "analysis/gulf.md")
    x, y, h, lon, lat = load_gulf()
    g = gauge(lon, lat, h)
    distance = float(np.hypot((lon[g[1]] - SPB[0]) * 111195 * np.cos(np.deg2rad(60)),
                              (lat[g[0]] - SPB[1]) * 111195))
    t = np.linspace(0, 12 * 3600, 721)
    scenarios = [("rest", 0., 0.), ("wave_only", .5, 0.), ("wind_only", 0., 25.),
                 ("combined_20", .5, 20.), ("combined_25", .5, 25.), ("combined_30", .5, 30.)]
    storage = {"x": x, "y": y, "lon": lon, "lat": lat, "h": h, "t": t}
    records = []
    maps = {}
    for name, a, wind in scenarios:
        start = time.perf_counter()
        extra = {"wind_speed": wind, "wind_dir": 270., "friction": .0025, "coriolis": 1., "dp": 0.}
        frames, peak = solve_2d(x, y, h, np.zeros_like(h), np.zeros_like(h), "gulf", t,
                                a=a, wavelength=300000., extra=extra)
        if not np.isfinite(frames).all() or not np.isfinite(peak).all():
            raise ValueError("Nonfinite surge output")
        series = frames[:, g[0], g[1]]
        ratio = np.divide(peak, h, out=np.full_like(h, np.nan), where=h > 0)
        area = (x[1] - x[0]) * (y[1] - y[0])
        record = {"id": name, "input_amplitude_m": a, "wavelength_m": 300000., "extra": extra,
                  "peak_at_gauge_m": float(peak[g]),
                  "sampled_peak_at_gauge_m": float(series.max()),
                  "sampled_peak_time_s": float(t[np.argmax(series)]),
                  "zone_over_0_1_km2": float(np.sum(ratio > .1) * area / 1e6),
                  "zone_over_0_3_km2": float(np.sum(ratio > .3) * area / 1e6),
                  "elapsed_s": time.perf_counter() - start}
        if name == "combined_25":
            fine_t = np.linspace(0, t[-1], 1441)
            fine_frames, fine_peak = solve_2d(x, y, h, np.zeros_like(h), np.zeros_like(h), "gulf", fine_t,
                                              a=a, wavelength=300000., extra=extra)
            record["frame_refinement_peak_change_m"] = float(abs(fine_peak[g] - peak[g]))
            storage["combined_25_fine_gauge"] = fine_frames[:, g[0], g[1]]
            storage["fine_t"] = fine_t
        records.append(record)
        storage[f"{name}_gauge"] = series
        storage[f"{name}_eta_max"] = peak
        maps[name] = ratio
        print(f"surge {name}: peak={peak[g]:.4f} m; {record['elapsed_s']:.1f} s", flush=True)
    np.savez_compressed(out / "third_surge.npz", **storage)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    labels_scenarios = {'wave_only': 'Волна без ветра', 'wind_only': 'Только ветер 25 м/с',
                       **{f'combined_{w}': f'Волна + ветер {w} м/с' for w in (20, 25, 30)}}
    for record in records:
        if record["id"] != "rest":
            axes[0].plot(t / 3600, storage[record['id'] + '_gauge'], label=labels_scenarios[record['id']])
    axes[0].set(xlabel="Время модельного опыта, ч", ylabel="Подъём у расчётной точки, м",
                title="Модельные сценарии: постоянный западный ветер")
    axes[0].legend(fontsize=8)
    levels = [r["reported_level_m"] for r in observations]
    labels = [re.search(r"\b(?:17|18|19|20)\d{2}\b", r['event'])[0] for r in observations]
    axes[1].bar(np.arange(len(levels)), levels, color=".65", label="Опубликованный максимум")
    combined = [r['peak_at_gauge_m'] for r in records if r['id'].startswith('combined')]
    axes[1].axhspan(min(combined), max(combined), alpha=.2, color="steelblue", label="Модель: ветер 20–30 м/с")
    axes[1].set(xticks=np.arange(len(labels)), xticklabels=labels, ylabel="Опубликованный уровень / модельный подъём, м",
                title="Сопоставление масштаба; нули не согласованы")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(axis="y", alpha=.2)
    fig.savefig(figures / "third_surge_observations.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), constrained_layout=True)
    for ax, name, title in zip(axes, ("wave_only", "combined_25"), ("Волна 0,5 м без ветра", "Волна + западный ветер 25 м/с")):
        ax.pcolormesh(lon, lat, np.ma.masked_where(h <= 0, h), cmap="Greys", alpha=.3, shading="auto")
        image = ax.pcolormesh(lon, lat, np.ma.masked_where(h <= 0, storage[name + '_eta_max']),
                              shading="auto", cmap="Blues", vmin=0, vmax=max(combined))
        cs = ax.contour(lon, lat, maps[name], levels=[.1, .3], colors=["orange", "purple"], linewidths=1)
        ax.clabel(cs, fmt={.1: "ηmax/h=0.1", .3: "ηmax/h=0.3"}, fontsize=7)
        ax.plot(lon[g[1]], lat[g[0]], "r*", ms=8, label="Расчётная точка у Петербурга")
        ax.set(title=title, xlabel="Долгота, ° в. д.", ylabel="Широта, ° с. ш.")
        ax.set_aspect(2)
        ax.legend(fontsize=7, loc="lower right")
    fig.colorbar(image, ax=axes, label="Максимальный подъём за опыт, м")
    fig.savefig(figures / "third_surge_zone.png", dpi=160)
    plt.close(fig)
    report = {"records": records, "observations": observations, "gauge_index_yx": [int(g[0]), int(g[1])],
              "gauge_lonlat": [float(lon[g[1]]), float(lat[g[0]])], "target_lonlat": list(SPB),
              "distance_to_target_m": distance, "gauge_depth_m": float(h[g]),
              "comparison": "scale comparison only; historical datums not transformed; no fitted storm forcing",
              "peak_method": "eta_max over internal solver steps; gauge curves sampled each 60 s",
              "zone_method": "local positive eta_max/h indicator, not a linear-vs-nonlinear difference"}
    write_json(out / "third_surge.json", report)
    return report


def audit_results(out, figures=None):
    """Check stored experiment data and enrich dispersion threshold records.

    Can also be run after separate sections; does not repeat numerical solves.
    """
    checks = {}
    path = out / "third_dispersion.json"
    if path.exists():
        report = json.loads(path.read_text(encoding="utf-8"))
        with np.load(out / "third_dispersion.npz", allow_pickle=False) as arrays:
            for i, record in enumerate(report["records"]):
                t = arrays[f"case_{i}_t"]
                record['kdv_vs_boussinesq'], _ = pair_metrics(
                    arrays[f'case_{i}_kdv_final'], arrays[f'case_{i}_boussinesq_final'], record['params']['a'])
                record['final_peak_over_a'] = {m: float(arrays[f'case_{i}_{m}_final'].max() / record['params']['a'])
                                                for m in MODEL_IDS}
                for model in ("kdv", "boussinesq"):
                    d = arrays[f"case_{i}_{model}_d_from_sv"]
                    first, previous = first_exceedance(t, d)
                    record["comparisons_to_saint_venant"][model].update(
                        t10_s=first, t10_previous_frame_s=previous,
                        t10_over_tstar=first / record['t_star_s'] if first is not None else None)
                    if not np.isfinite(d).all() or d[0] > 1e-9:
                        raise ValueError("Invalid dispersion diagnostic")
                if record['kh'] > .3 or record['epsilon'] >= .01:
                    raise ValueError("Weak shallow-wave experiment left its declared regime")
        write_json(path, report)
        checks['dispersion'] = {'cases': len(report['records']), 'finite_diagnostics': True,
                                'identical_initial_surface': True, 'all_kh_below_0_3': True}
    path = out / "third_gulf.json"
    if path.exists():
        report = json.loads(path.read_text(encoding="utf-8"))
        if len(report['records']) != 8:
            raise ValueError("Incomplete gulf amplitude sweep")
        if any(r['last_included_frame_s'] >= r['common_cutoff_s'] for r in report['records']):
            raise ValueError('Gulf diagnostic includes coastal boundary response')
        checks['gulf'] = {'cases': 8, 'pre_boundary_window': True,
                          'numerical_limitations': 'See refinement errors and control_error in third_gulf.json'}
        with np.load(out / 'third_gulf.npz', allow_pickle=False) as arrays:
            if any(np.isinf(arrays[k]).any() for k in arrays.files):
                raise ValueError('Infinite gulf result')
        path_sampling = out / 'third_gulf_sampling.json'
        if path_sampling.exists():
            sampling = json.loads(path_sampling.read_text(encoding='utf-8'))
            checks['gulf']['output_sampling_control'] = sampling['records']
            checks['gulf']['cadence_peak_change_exceeds_0_1a'] = bool(
                sampling['records']['boussinesq']['peak_linf_change_over_a'] > .1)
            checks['gulf']['boussinesq_envelope_status'] = 'convergence not claimed; inspect spatial and cadence errors'
    path = out / "third_surge.json"
    if path.exists():
        report = json.loads(path.read_text(encoding="utf-8"))
        if len(report['records']) != 6 or len(report['observations']) != 10:
            raise ValueError("Incomplete surge comparison")
        with np.load(out / "third_surge.npz", allow_pickle=False) as arrays:
            for record in report['records']:
                curve = arrays[record['id'] + '_gauge']
                if not np.isfinite(curve).all() or curve.max() > record['peak_at_gauge_m'] + 1e-10:
                    raise ValueError("Invalid surge peak")
            if np.max(abs(arrays['rest_eta_max'])) > 1e-10:
                raise ValueError("Lake-at-rest control failed")
        checks['surge'] = {'scenarios': 6, 'sourced_observations': 10, 'rest_control': True,
                           'historical_validation': 'scale only; storm forcing and datum reconciliation unavailable'}
    checks['all_sections_present'] = all(k in checks for k in ('dispersion', 'gulf', 'surge')) and (out / 'third_gulf_sampling.json').exists()
    if figures is not None:
        names = ['third_dispersion_profiles.png', 'third_dispersion_metrics.png', 'third_gulf_envelopes.png',
                 'third_surge_observations.png', 'third_surge_zone.png']
        checks['figures_present'] = [n for n in names if (figures / n).is_file() and (figures / n).stat().st_size > 0]
    write_json(out / "third_checks.json", checks)
    return checks


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--section", choices=["all", "dispersion", "gulf", "sampling", "surge", "audit"], default="all")
    parser.add_argument("--dispersion-nx", type=int, default=1024)
    parser.add_argument("--gulf-nx", type=int, default=3201)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "analysis/results")
    parser.add_argument("--figures-dir", type=Path, default=ROOT / "analysis/figures")
    args = parser.parse_args(argv)
    if args.dispersion_nx < 64 or args.dispersion_nx % 4 or args.gulf_nx < 101:
        parser.error("dispersion nx must be >=64 and divisible by 4; gulf nx must be >=101")
    for folder in (args.results_dir, args.figures_dir):
        folder.mkdir(parents=True, exist_ok=True)
    if args.section in ("all", "dispersion"):
        study_dispersion(args.results_dir, args.figures_dir, args.dispersion_nx)
    if args.section in ("all", "gulf"):
        study_gulf(args.results_dir, args.figures_dir, args.gulf_nx)
    if args.section in ("all", "gulf", "sampling"):
        study_gulf_sampling(args.results_dir, args.gulf_nx)
    if args.section in ("all", "surge"):
        study_surge(args.results_dir, args.figures_dir)
    audit_results(args.results_dir, args.figures_dir)
    provenance = {}
    for relative in ["analysis/third_wave.py", "analysis/compare.py", "backend/contract.py", "backend/registry.py",
                     "backend/common.py", "backend/models/linear.py", "backend/models/saint_venant.py", "backend/models/kdv.py",
                     "backend/models/boussinesq.py", "backend/models/shallow_2d.py",
                     "data/gulf_profile.npz", "data/gulf_bathymetry.npz", "analysis/gulf.md",
                     "analysis/results/compare_summary.json"]:
        provenance[relative] = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
    write_json(args.results_dir / "third_provenance.json", provenance)


if __name__ == "__main__":
    main()
