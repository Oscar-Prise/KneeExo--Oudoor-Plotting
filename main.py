import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

DATA_FILE = "torque_pilot.npz"

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

d = np.load(DATA_FILE)
t = d["time"]

# Activity segments in data time (s). The video runs ~2.8 s ahead of the data;
# boundaries were set from stride period, knee flexion depth, and turnaround pauses.
# Turnarounds (19.2-21.0, 44.8-46.4) are included in the preceding bout.
SEGMENTS = [
    ("Squat", 6.3, 11.0),
    ("RD slow", 11.0, 15.9),
    ("RD fast", 15.9, 21.0),
    ("RA slow", 21.0, 25.9),
    ("RA fast", 25.9, 29.1),
    ("LG slow", 29.1, 35.0),
    ("LG fast", 35.0, 40.6),
    ("SA", 40.6, 46.4),
    ("SD", 46.4, 50.8),
]
# Time window shown in figures 1 and 2: start of RD slow to end of SD.
TRIAL_START, TRIAL_END = SEGMENTS[1][1], SEGMENTS[-1][2]
# Turnaround windows: strides overlapping these are left out of gait-phase averages.
TURNAROUNDS = [(19.2, 21.0), (44.8, 46.4)]
# One hue per activity; fast bouts use a stronger tint than slow ones.
ACTIVITY_COLOR = {
    "Squat": "#4a3aa7",  # violet
    "RD": "#eb6834",     # orange
    "RA": "#1baf7a",     # aqua
    "LG": "#eda100",     # yellow
    "SA": "#e87ba4",     # magenta
    "SD": "#008300",     # green
}

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


def walking_time(start, end):
    """Bout duration with any turnaround time removed."""
    turn = sum(max(0, min(end, tb) - max(start, ta)) for ta, tb in TURNAROUNDS)
    return end - start - turn


BOUT_SPEED = {name: BOUT_DISTANCE_M[name] / walking_time(start, end)
              for name, start, end in SEGMENTS if name in BOUT_DISTANCE_M}


def shade_segments(axes, label_ax):
    for name, start, end in SEGMENTS:
        activity = name.split()[0]
        alpha = 0.30 if name.endswith("fast") else 0.15
        for ax in axes:
            ax.axvspan(start, end, color=ACTIVITY_COLOR[activity], alpha=alpha, zorder=0, lw=0)
        if start < TRIAL_START or end > TRIAL_END:
            continue
        label_ax.text((start + end) / 2, 1.0, f"{name}\n{BOUT_SPEED[name]:.2f} m/s",
                      transform=label_ax.get_xaxis_transform(),
                      ha="center", va="bottom", fontsize=8, color=INK, linespacing=1.3)


def detect_heel_strikes(t, gyro, t_start, t_end):
    """Heel strike = first positive-to-negative zero crossing after each swing peak.

    In swing the foot gyro (z) forms a positive hump; at initial contact it drops
    sharply through zero into the negative foot-slap dip.
    """
    swing_peaks, _ = find_peaks(gyro, height=100, prominence=150, distance=50)
    hs = []
    for p in swing_peaks:
        if not t_start <= t[p] <= t_end:
            continue
        after = np.nonzero(gyro[p:] < 0)[0]
        if after.size:
            hs.append(t[p + after[0]])
    return np.array(hs)


heel_strikes = detect_heel_strikes(t, d["gyro_foot_r_z"], SEGMENTS[1][1], SEGMENTS[-1][2])

PHASE = np.linspace(0, 100, 101)  # gait phase, % of stride


def strides_by_phase(t, signal, hs, start, end):
    """Resample each heel-strike-to-heel-strike stride inside [start, end] onto PHASE.

    Strides that cross a bout boundary or overlap a turnaround are skipped.
    """
    strides = []
    for hs0, hs1 in zip(hs[:-1], hs[1:]):
        if hs0 < start or hs1 > end:
            continue
        if any(hs0 < tb and hs1 > ta for ta, tb in TURNAROUNDS):
            continue
        m = (t >= hs0) & (t <= hs1)
        phase = (t[m] - hs0) / (hs1 - hs0) * 100
        strides.append(np.interp(PHASE, phase, signal[m]))
    return np.array(strides)

# Figure 1: commanded torque, left and right
fig1, ax = plt.subplots(figsize=(11, 4))
ax.plot(t, d["cmd_R"], color=COLOR_R, label="Right")
ax.plot(t, d["cmd_L"], color=COLOR_L, label="Left")
ax.set_title("Commanded torque", pad=28)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Torque command (N·m)")
ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
shade_segments([ax], ax)
ax.set_xlim(TRIAL_START, TRIAL_END)
fig1.tight_layout()

# Figure 2: right torque and knee angle, shared time axis
fig2, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)

axes[0].plot(t, d["cmd_R"], color=COLOR_R)
axes[0].set_title("Right commanded torque")
axes[0].set_ylabel("N·m")

axes[1].plot(t, d["knee_angle_r"], color=COLOR_R)
axes[1].set_title("Right knee angle (negative = flexion)")
axes[1].set_ylabel("deg")
axes[1].set_xlabel("Time (s)")

shade_segments(axes, axes[0])
for ax in axes:
    for hs in heel_strikes:
        ax.axvline(hs, color=INK, ls="--", lw=0.7, zorder=1)
axes[0].set_title("Right commanded torque", pad=28)
axes[0].set_xlim(TRIAL_START, TRIAL_END)
fig2.tight_layout()

# Figure 3: right commanded torque over gait phase, mean ± SD per walking bout
walking_bouts = [s for s in SEGMENTS if s[0] != "Squat"]
fig3, axes = plt.subplots(2, 4, figsize=(14, 6.5), sharex=True, sharey=True)
for ax, (name, start, end) in zip(axes.flat, walking_bouts):
    strides = strides_by_phase(t, d["cmd_R"], heel_strikes, start, end)
    mean, sd = strides.mean(axis=0), strides.std(axis=0)
    ax.fill_between(PHASE, mean - sd, mean + sd, color=COLOR_R, alpha=0.2, lw=0)
    ax.plot(PHASE, mean, color=COLOR_R, lw=1.6)
    ax.axhline(0, color=INK, lw=0.6)
    ax.set_title(f"{name}  ·  {BOUT_SPEED[name]:.2f} m/s  (n = {len(strides)} strides)")
    ax.set_xlim(0, 100)
for ax in axes[1]:
    ax.set_xlabel("Gait phase (% stride, HS to HS)")
# Positive torque assists knee extension (it is positive while rising out of each squat).
for ax in axes[:, 0]:
    ax.set_ylabel("Flexion  ←  Torque (N·m)  →  Extension")
fig3.suptitle("Right commanded torque over gait phase (mean ± SD)", x=0.01, ha="left",
              fontsize=12, color=INK)
fig3.tight_layout()

plt.show()
