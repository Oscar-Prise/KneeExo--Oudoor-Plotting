import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Per-subject data file and activity segments in data time (s). Video times are logged
# in bout_timing_video.txt; boundaries here were refined from stride period, knee flexion
# depth, and turnaround pauses. Turnarounds are included in the preceding bout; strides
# overlapping them are left out of gait-phase averages and their time is left out of
# bout speeds.
SUBJECTS = {
    # Data runs ~2.8 s behind the video.
    "AB01": {
        "file": "AB01_Itak.npz",
        "segments": [
            ("Squat", 6.3, 11.0),
            ("RD slow", 11.0, 15.9),
            ("RD fast", 15.9, 21.0),
            ("RA slow", 21.0, 25.9),
            ("RA fast", 25.9, 29.1),
            ("LG slow", 29.1, 35.0),
            ("LG fast", 35.0, 40.6),
            ("SA", 40.6, 46.4),
            ("SD", 46.4, 50.8),
        ],
        "turnarounds": [(19.5, 21.0), (44.8, 46.4)],
    },
    # Data runs ~14.3 s behind the video.
    "AB02": {
        "file": "AB02_Oscar.npz",
        "segments": [
            ("Squat", 15.6, 20.0),
            ("RD slow", 20.0, 23.6),
            ("RD fast", 23.6, 27.5),
            ("RA slow", 27.5, 31.4),
            ("RA fast", 31.4, 34.2),
            ("LG slow", 34.2, 39.0),
            ("LG fast", 39.0, 44.4),
            ("SA", 44.4, 49.4),
            ("SD", 49.4, 53.1),
        ],
        "turnarounds": [(26.3, 27.5), (48.4, 49.4)],
    },
    # Data runs ~6 s behind the video.
    "AB03": {
        "file": "AB03_Rajiv.npz",
        "segments": [
            ("Squat", 5.6, 9.8),
            ("RD slow", 9.8, 14.7),
            ("RD fast", 14.7, 20.0),
            ("RA slow", 20.0, 23.7),
            ("RA fast", 23.7, 27.0),
            ("LG slow", 27.0, 31.3),
            ("LG fast", 31.3, 36.2),
            ("SA", 36.2, 41.5),
            ("SD", 41.5, 45.6),
        ],
        "turnarounds": [(18.0, 20.0), (40.5, 41.5)],
    },
    # Data runs ~13.3 s behind the video.
    "AB04": {
        "file": "AB04_Changseob.npz",
        "segments": [
            ("Squat", 14.3, 20.3),
            ("RD slow", 20.3, 25.4),
            ("RD fast", 25.4, 29.3),
            ("RA slow", 29.3, 34.1),
            ("RA fast", 34.1, 37.0),
            ("LG slow", 37.0, 41.7),
            ("LG fast", 41.7, 47.0),
            ("SA", 47.0, 55.2),
            ("SD", 55.2, 60.3),
        ],
        "turnarounds": [(28.3, 29.3), (52.8, 55.2)],
    },
}

TIMING_FILE = "bout_timing_video.txt"
FIG_DIR = "figures"

COLOR_R = "#2a78d6"  # blue
COLOR_L = "#eb6834"  # orange
INK = "#3d3d3a"
GRID = "#e4e3dc"

plt.rcParams.update({
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "axes.titlesize": 11,
    "axes.titlelocation": "left",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": INK,
    "ytick.color": INK,
    "lines.linewidth": 1.2,
    "legend.frameon": False,
})

# Distance walked per bout (m). Stairs: 7 steps along the diagonal of a 30 cm x 15 cm step.
FT_IN = 0.3048, 0.0254
RAMP_M = 17 * FT_IN[0] + 8 * FT_IN[1]
STAIRS_M = 7 * np.hypot(0.30, 0.15)
BOUT_DISTANCE_M = {
    "RD slow": RAMP_M,
    "RD fast": RAMP_M,
    "RA slow": RAMP_M,
    "RA fast": RAMP_M,
    "LG slow": 18 * FT_IN[0] + 9 * FT_IN[1],
    "LG fast": 33 * FT_IN[0] + 2 * FT_IN[1],
    "SA": STAIRS_M,
    "SD": STAIRS_M,
}
WALKING_BOUTS = list(BOUT_DISTANCE_M)

# One hue per activity; fast bouts use a stronger tint than slow ones.
ACTIVITY_COLOR = {
    "Squat": "#4a3aa7",  # violet
    "RD": "#eb6834",     # orange
    "RA": "#1baf7a",     # aqua
    "LG": "#eda100",     # yellow
    "SA": "#e87ba4",     # magenta
    "SD": "#008300",     # green
}

PHASE = np.linspace(0, 100, 101)  # gait phase, % of stride
FLEX_EXT_LABEL = "Flexion  ←  Torque (N·m)  →  Extension"


def read_subject_info(path):
    """Weight/height/age per subject ID (e.g. "AB01") from the [ABxx_Name] sections."""
    info, subj = {}, None
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.split("#")[0].strip()
            if line.startswith("["):
                subj = line.strip("[]").split("_")[0]
                info[subj] = {}
            elif subj and ":" in line:
                key, value = line.split(":", 1)
                info[subj][key.strip().lower()] = value.strip()
    return info


def info_label(info):
    """e.g. "72.4 kg · 179 cm · 24 yr" """
    parts = [info.get("weight"), info.get("height")]
    if "age" in info:
        parts.append(f"{info['age']} yr")
    return "  ·  ".join(p for p in parts if p)


def walking_time(start, end, turnarounds):
    """Bout duration with any turnaround time removed."""
    turn = sum(max(0, min(end, tb) - max(start, ta)) for ta, tb in turnarounds)
    return end - start - turn


def detect_heel_strikes(t, gyro, t_start, t_end):
    """Heel strike = first positive-to-negative zero crossing after each swing peak.

    In swing the foot gyro (z) forms a positive hump; at initial contact it drops
    sharply through zero into the negative foot-slap dip.
    """
    # height=50 (not 100) so gentle stair-ascent swings (e.g. AB04's ~98 deg/s) still count.
    swing_peaks, _ = find_peaks(gyro, height=50, prominence=150, distance=50)
    hs = []
    for p in swing_peaks:
        if not t_start <= t[p] <= t_end:
            continue
        after = np.nonzero(gyro[p:] < 0)[0]
        if after.size:
            hs.append(t[p + after[0]])
    return np.array(hs)


def strides_by_phase(t, signal, hs, start, end, turnarounds):
    """Resample each heel-strike-to-heel-strike stride inside [start, end] onto PHASE.

    Strides that cross a bout boundary or overlap a turnaround are skipped.
    """
    strides = []
    for hs0, hs1 in zip(hs[:-1], hs[1:]):
        if hs0 < start or hs1 > end:
            continue
        if any(hs0 < tb and hs1 > ta for ta, tb in turnarounds):
            continue
        m = (t >= hs0) & (t <= hs1)
        phase = (t[m] - hs0) / (hs1 - hs0) * 100
        strides.append(np.interp(PHASE, phase, signal[m]))
    return np.array(strides)


def process_subject(cfg):
    """Raw data, heel strikes, per-bout mean stride curves, and bout speeds."""
    d = np.load(cfg["file"])
    t = d["time"]
    segments = {name: (start, end) for name, start, end in cfg["segments"]}
    trial = segments["RD slow"][0], segments["SD"][1]
    hs = detect_heel_strikes(t, d["gyro_foot_r_z"], *trial)
    # Foot pivots while turning around look like small swings; don't count them.
    hs = np.array([h for h in hs if not any(ta <= h <= tb for ta, tb in cfg["turnarounds"])])

    out = {"d": d, "t": t, "hs": hs, "trial": trial, "segments": cfg["segments"],
           "stride_mean": {}, "n_strides": {}, "speed": {}}
    for name in WALKING_BOUTS:
        start, end = segments[name]
        strides = strides_by_phase(t, d["cmd_R"], hs, start, end, cfg["turnarounds"])
        out["n_strides"][name] = len(strides)
        if len(strides):
            out["stride_mean"][name] = strides.mean(axis=0)
        out["speed"][name] = BOUT_DISTANCE_M[name] / walking_time(start, end, cfg["turnarounds"])
    return out


subject_info = read_subject_info(TIMING_FILE)
results = {subj: process_subject(cfg) for subj, cfg in SUBJECTS.items()}

print("Bout speed (m/s) and strides per subject")
print(f"{'':8s}" + "".join(f"{s:>14s}" for s in results) + f"{'mean +/- SD':>16s}")
for name in WALKING_BOUTS:
    speeds = np.array([r["speed"][name] for r in results.values()])
    cells = "".join(f"{r['speed'][name]:8.2f} (n={r['n_strides'][name]})" for r in results.values())
    print(f"{name:8s}{cells}{speeds.mean():10.2f} +/- {speeds.std():.2f}")


def shade_segments(axes, label_ax, r):
    """Bout shading on every axis; bout name + speed above label_ax."""
    trial_start, trial_end = r["trial"]
    for name, start, end in r["segments"]:
        activity = name.split()[0]
        alpha = 0.30 if name.endswith("fast") else 0.15
        for ax in axes:
            ax.axvspan(start, end, color=ACTIVITY_COLOR[activity], alpha=alpha, zorder=0, lw=0)
        if start < trial_start or end > trial_end:
            continue
        label_ax.text((start + end) / 2, 1.0, f"{name}\n{r['speed'][name]:.2f} m/s",
                      transform=label_ax.get_xaxis_transform(),
                      ha="center", va="bottom", fontsize=8, color=INK, linespacing=1.3)


def subject_title(subj, what):
    return f"{subj}  ({info_label(subject_info.get(subj, {}))})  ·  {what}"


def save_png(fig, path):
    # On Windows a PNG that is open in an image viewer (e.g. Photos) can't be overwritten;
    # Python reports that as the unhelpful "[Errno 22] Invalid argument".
    try:
        fig.savefig(path, dpi=200)
    except OSError as e:
        raise OSError(f"Couldn't write {path}. Close it in any image viewer (e.g. Photos) "
                      f"and run again.") from e


# Per-subject time plots, saved as PNG (not shown).
os.makedirs(FIG_DIR, exist_ok=True)
for subj, r in results.items():
    d, t = r["d"], r["t"]

    # Left and right commanded torque
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(t, d["cmd_R"], color=COLOR_R, label="Right")
    ax.plot(t, d["cmd_L"], color=COLOR_L, label="Left")
    ax.set_title(subject_title(subj, "Commanded torque"), pad=28)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(FLEX_EXT_LABEL)
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    shade_segments([ax], ax, r)
    ax.set_xlim(*r["trial"])
    fig.tight_layout()
    save_png(fig, os.path.join(FIG_DIR, f"{subj}_torque_LR.png"))
    plt.close(fig)

    # Right torque and knee angle with right heel strikes
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    axes[0].plot(t, d["cmd_R"], color=COLOR_R)
    axes[0].set_title(subject_title(subj, "Right commanded torque"), pad=28)
    axes[0].set_ylabel(FLEX_EXT_LABEL)
    axes[1].plot(t, d["knee_angle_r"], color=COLOR_R)
    axes[1].set_title("Right knee angle (negative = flexion)")
    axes[1].set_ylabel("deg")
    axes[1].set_xlabel("Time (s)  ·  dashed lines = right heel strike")
    shade_segments(axes, axes[0], r)
    for ax in axes:
        for hs in r["hs"]:
            ax.axvline(hs, color=INK, ls="--", lw=0.7, zorder=1)
    axes[0].set_xlim(*r["trial"])
    fig.tight_layout()
    save_png(fig, os.path.join(FIG_DIR, f"{subj}_torque_R_HS.png"))
    plt.close(fig)

print(f"Saved per-subject figures to {FIG_DIR}/")


def plot_mean_sd(ax, x, curves, color):
    curves = np.array(curves)
    mean, sd = curves.mean(axis=0), curves.std(axis=0)
    ax.fill_between(x, mean - sd, mean + sd, color=color, alpha=0.2, lw=0)
    ax.plot(x, mean, color=color, lw=1.4)


# Right commanded torque over gait phase per walking bout. Each subject's strides are
# averaged first, then mean ± SD is taken across subjects.
fig3, axes = plt.subplots(2, 4, figsize=(14, 6.5), sharex=True, sharey=True)
for ax, name in zip(axes.flat, WALKING_BOUTS):
    curves = [r["stride_mean"][name] for r in results.values() if name in r["stride_mean"]]
    speeds = np.array([r["speed"][name] for r in results.values()])
    n_strides = sum(r["n_strides"][name] for r in results.values())
    plot_mean_sd(ax, PHASE, curves, COLOR_R)
    ax.axhline(0, color=INK, lw=0.6)
    ax.set_title(f"{name}  ·  {speeds.mean():.2f} ± {speeds.std():.2f} m/s\n"
                 f"n = {len(curves)} subjects, {n_strides} strides", fontsize=10)
    ax.set_xlim(0, 100)
for ax in axes[1]:
    ax.set_xlabel("Gait phase (% stride, HS to HS)")
# Positive torque assists knee extension (it is positive while rising out of each squat).
for ax in axes[:, 0]:
    ax.set_ylabel(FLEX_EXT_LABEL)
fig3.suptitle(f"Right commanded torque over gait phase (mean ± SD across {', '.join(results)})",
              x=0.01, ha="left", fontsize=12, color=INK)
fig3.tight_layout()
save_png(fig3, "torque_gait_phase_all.png")

plt.show()
