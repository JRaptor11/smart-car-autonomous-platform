from typing import List, Tuple, Dict, Optional, Any

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _to_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ============================================================
# ==================== STYLE / MODES =========================
# ============================================================

FIGSIZE = (10, 6)

# ---- Layering ----
REF_ZORDER = 5
CAR_ZORDER = 3
TGT_ZORDER = 1

# ---- Car trajectory style ----
CAR_COLOR = "tab:blue"
CAR_LINESTYLE = "-"
CAR_LINEWIDTH = 1.5
CAR_ALPHA = 1.0

# ---- Reference path style ----
REF_COLOR = "tab:orange"
REF_LINESTYLE = "--"
REF_LINEWIDTH = 1.5
REF_ALPHA = 1.0

# ---- Target point style ----
TGT_COLOR = "olive"
TGT_EDGE_COLOR = "black"
TGT_EDGE_WIDTH = 0.4

TARGET_MODE = "all"   # "unique", "all", or "time"
AUTO_STYLE_BY_MODE = True
TGT_MARKER_SIZE = 18
TGT_ALPHA = 0.90
TIME_CMAP = "viridis"

# ---- Cropping controls ----
REF_MARGIN_PTS = 10     # show a little path ahead of where the run reached
X_MARGIN_LEFT = 1.0
X_MARGIN_RIGHT = 3.0
Y_MARGIN = 1.0

# ============================================================


if len(sys.argv) < 2:
    print("Usage: python plot_log.py path/to/log.csv")
    sys.exit(1)

csv_path = sys.argv[1]
df = pd.read_csv(csv_path)

df = _to_numeric(
    df,
    [
        "t",
        "x",
        "y",
        "ref_x",
        "ref_y",
        "ref_i",
        "ref_s_m",
        "tgt_x",
        "tgt_y",
        "tgt_i",
        "tgt_s_m",
    ],
)

# Separate seeded rows from real runtime rows
if "source" in df.columns:
    runtime_df = df[df["source"] != "REF_SEED"].copy()
else:
    runtime_df = df.copy()

plt.figure(figsize=FIGSIZE)

# -----------------------------
# Target points (bottom layer)
# -----------------------------
tgt_i_max = None
if "tgt_i" in runtime_df.columns:
    tgt_i_series = pd.to_numeric(runtime_df["tgt_i"], errors="coerce")
    if tgt_i_series.notna().any():
        tgt_i_max = int(tgt_i_series.max())

tgt_all = None
if {"tgt_x", "tgt_y"}.issubset(runtime_df.columns):
    tgt_all = runtime_df.dropna(subset=["tgt_x", "tgt_y"]).copy()

    if len(tgt_all) > 0:
        n_total = len(runtime_df)
        n_valid = len(tgt_all)
        n_unique = tgt_all.drop_duplicates(subset=["tgt_x", "tgt_y"]).shape[0]
        print(f"Target points: {n_valid}/{n_total} runtime rows have valid tgt_x/tgt_y")
        print(f"Target points unique positions: {n_unique}/{n_valid}")

        if AUTO_STYLE_BY_MODE:
            if TARGET_MODE == "unique":
                size = 25
                alpha = 0.95
                edge_w = 0.4
            elif TARGET_MODE == "all":
                size = 8
                alpha = 0.95
                edge_w = 0.0
            elif TARGET_MODE == "time":
                size = 18
                alpha = 0.95
                edge_w = 0.0
            else:
                size = 18
                alpha = 0.90
                edge_w = TGT_EDGE_WIDTH
        else:
            size = TGT_MARKER_SIZE
            alpha = TGT_ALPHA
            edge_w = TGT_EDGE_WIDTH

        if TARGET_MODE == "unique":
            tgt_plot = tgt_all.drop_duplicates(subset=["tgt_x", "tgt_y"], keep="first")
            plt.scatter(
                tgt_plot["tgt_x"],
                tgt_plot["tgt_y"],
                s=size,
                c=TGT_COLOR,
                edgecolors=TGT_EDGE_COLOR,
                linewidths=edge_w,
                alpha=alpha,
                zorder=TGT_ZORDER,
                label="Target points (unique)",
            )

        elif TARGET_MODE == "all":
            plt.scatter(
                tgt_all["tgt_x"],
                tgt_all["tgt_y"],
                s=size,
                c=TGT_COLOR,
                edgecolors=TGT_EDGE_COLOR,
                linewidths=edge_w,
                alpha=alpha,
                zorder=TGT_ZORDER,
                label="Target points (all)",
            )

        elif TARGET_MODE == "time":
            if "t" in tgt_all.columns and pd.to_numeric(tgt_all["t"], errors="coerce").notna().any():
                tvals = pd.to_numeric(tgt_all["t"], errors="coerce").interpolate(limit_direction="both")
            else:
                tvals = np.arange(len(tgt_all), dtype=float)

            sc = plt.scatter(
                tgt_all["tgt_x"],
                tgt_all["tgt_y"],
                s=size,
                c=tvals,
                cmap=TIME_CMAP,
                edgecolors="none",
                alpha=alpha,
                zorder=TGT_ZORDER,
                label="Target points (time-colored)",
            )
            cbar = plt.colorbar(sc)
            cbar.set_label("time" if "t" in tgt_all.columns else "index")

        else:
            print(f"[WARN] Unknown TARGET_MODE='{TARGET_MODE}'. Use 'unique', 'all', or 'time'.")

missing = runtime_df[runtime_df["tgt_x"].isna() | runtime_df["tgt_y"].isna()].copy()
print(f"Missing tgt rows: {len(missing)}/{len(runtime_df)}")

if "source" in runtime_df.columns:
    print("Missing by source:")
    print(missing["source"].value_counts(dropna=False))

if "cmd_valid" in runtime_df.columns:
    print("Missing by cmd_valid:")
    print(missing["cmd_valid"].value_counts(dropna=False))

cols = [c for c in ["t", "source", "cmd_valid", "ref_i", "tgt_i", "x", "y"] if c in runtime_df.columns]
print(missing[cols].head(10))

# -----------------------------
# Car trajectory (middle layer)
# -----------------------------
traj = runtime_df.dropna(subset=["x", "y"]).copy()
plt.plot(
    traj["x"],
    traj["y"],
    color=CAR_COLOR,
    linestyle=CAR_LINESTYLE,
    linewidth=CAR_LINEWIDTH,
    alpha=CAR_ALPHA,
    zorder=CAR_ZORDER,
    label="Car trajectory",
)

# -----------------------------
# Reference path logic
# -----------------------------
ref_plotted = False

# Limit reference path to only the relevant portion reached during the run
ref_i_limit = None

if "ref_i" in runtime_df.columns:
    ref_i_series = pd.to_numeric(runtime_df["ref_i"], errors="coerce")
    if ref_i_series.notna().any():
        ref_i_limit = int(ref_i_series.max())

if "tgt_i" in runtime_df.columns:
    tgt_i_series = pd.to_numeric(runtime_df["tgt_i"], errors="coerce")
    if tgt_i_series.notna().any():
        tgt_i_runtime_max = int(tgt_i_series.max())
        if ref_i_limit is None:
            ref_i_limit = tgt_i_runtime_max
        else:
            ref_i_limit = max(ref_i_limit, tgt_i_runtime_max)

if ref_i_limit is not None:
    ref_i_limit += REF_MARGIN_PTS

# Build reference curve from full df so REF_SEED is still available,
# but crop it using runtime-derived limit
ref_curve = None
if {"ref_i", "ref_x", "ref_y"}.issubset(df.columns):
    ref_pts = df.dropna(subset=["ref_i", "ref_x", "ref_y"]).copy()
    if len(ref_pts) > 0:
        ref_pts["ref_i"] = pd.to_numeric(ref_pts["ref_i"], errors="coerce")
        ref_pts = ref_pts.dropna(subset=["ref_i"])
        ref_pts["ref_i"] = ref_pts["ref_i"].astype(int)

        ref_pts = ref_pts.sort_values("ref_i").drop_duplicates(subset=["ref_i"], keep="first")
        ref_curve = ref_pts[["ref_i", "ref_x", "ref_y"]].copy()

# Extend using target points if needed
if {"tgt_i", "tgt_x", "tgt_y"}.issubset(df.columns):
    tgt_pts = df.dropna(subset=["tgt_i", "tgt_x", "tgt_y"]).copy()
    if len(tgt_pts) > 0:
        tgt_pts["tgt_i"] = pd.to_numeric(tgt_pts["tgt_i"], errors="coerce")
        tgt_pts = tgt_pts.dropna(subset=["tgt_i"])
        tgt_pts["tgt_i"] = tgt_pts["tgt_i"].astype(int)

        tgt_pts = tgt_pts.sort_values("tgt_i").drop_duplicates(subset=["tgt_i"], keep="first")

        ext = tgt_pts[["tgt_i", "tgt_x", "tgt_y"]].copy()
        ext.columns = ["ref_i", "ref_x", "ref_y"]

        if ref_curve is not None and len(ref_curve) > 0:
            ref_i_max_existing = int(ref_curve["ref_i"].max())
            ext = ext[ext["ref_i"] > ref_i_max_existing]

        if len(ext) > 0:
            if ref_curve is None:
                ref_curve = ext
            else:
                ref_curve = pd.concat([ref_curve, ext], ignore_index=True)

# Final reference plot
if ref_curve is not None and len(ref_curve) > 1:
    ref_curve = ref_curve.sort_values("ref_i").drop_duplicates(subset=["ref_i"], keep="first")

    if ref_i_limit is not None:
        ref_curve = ref_curve[ref_curve["ref_i"] <= ref_i_limit]

    plt.plot(
        ref_curve["ref_x"],
        ref_curve["ref_y"],
        color=REF_COLOR,
        linestyle=REF_LINESTYLE,
        linewidth=REF_LINEWIDTH,
        alpha=REF_ALPHA,
        zorder=REF_ZORDER,
        label="Reference path (ref_i)",
    )
    ref_plotted = True

# Fallback if needed
if (not ref_plotted) and {"ref_x", "ref_y"}.issubset(runtime_df.columns):
    ref_pts = runtime_df.dropna(subset=["ref_x", "ref_y"]).drop_duplicates(subset=["ref_x", "ref_y"])
    if len(ref_pts) > 0:
        ref_pts = ref_pts.sort_values("ref_x")
        plt.plot(
            ref_pts["ref_x"],
            ref_pts["ref_y"],
            color=REF_COLOR,
            linestyle=REF_LINESTYLE,
            linewidth=REF_LINEWIDTH,
            alpha=REF_ALPHA,
            zorder=REF_ZORDER,
            label="Reference path (approx)",
        )
        ref_plotted = True

# -----------------------------
# Titles / labels
# -----------------------------
title_bits = []
if tgt_i_max is not None:
    title_bits.append(f"max tgt_i={tgt_i_max}")
if ("ref_s_m" in df.columns) or ("tgt_s_m" in df.columns):
    title_bits.append("using s_m")

if len(title_bits) > 0:
    plt.title("Vehicle Trajectory vs Reference Path (" + ", ".join(title_bits) + ")")
else:
    plt.title("Vehicle Trajectory vs Reference Path")

plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.legend()
plt.grid(True)
plt.axis("equal")

# Crop x-axis to traveled region
if len(traj) > 0:
    x_min = float(traj["x"].min()) - X_MARGIN_LEFT
    x_max = float(traj["x"].max()) + X_MARGIN_RIGHT
    plt.xlim(x_min, x_max)

    y_min = float(min(traj["y"].min(), ref_curve["ref_y"].min() if ref_curve is not None and len(ref_curve) > 0 else traj["y"].min())) - Y_MARGIN
    y_max = float(max(traj["y"].max(), ref_curve["ref_y"].max() if ref_curve is not None and len(ref_curve) > 0 else traj["y"].max())) + Y_MARGIN
    plt.ylim(y_min, y_max)
else:
    if "y" in df.columns:
        yvals = pd.to_numeric(df["y"], errors="coerce").dropna()
        if len(yvals) > 0:
            plt.ylim(float(yvals.min()) - Y_MARGIN, float(yvals.max()) + Y_MARGIN)

plt.gca().set_aspect("equal", adjustable="box")
plt.show()