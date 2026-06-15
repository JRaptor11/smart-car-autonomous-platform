import sys
from typing import List

import pandas as pd
import matplotlib.pyplot as plt


def _to_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _has_any(df: pd.DataFrame, cols: List[str]) -> List[str]:
    present = []
    for c in cols:
        if c in df.columns and df[c].notna().any():
            present.append(c)
    return present


def _plot_lines(df: pd.DataFrame, xcol: str, ycols: List[str], title: str, ylabel: str) -> bool:
    ycols = _has_any(df, ycols)
    if not ycols:
        return False

    plt.figure(figsize=(10, 4))
    for c in ycols:
        plt.plot(df[xcol], df[c], label=c)

    plt.xlabel("time (s)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    return True


def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_debug_timeseries.py <path/to/log.csv>")
        sys.exit(1)

    csv_path = sys.argv[1]
    df = pd.read_csv(csv_path)

    # Convert useful fields to numeric when present
    numeric_cols = [
        "t",
        "lane_conf",
        "failsafe",
        "bad_frames",
        "steer_pwm_us",
        "throttle_pwm_us",
        "steer_rad",
        "accel_cmd",
        "cte",
        "heading_err",
        "cte_term",
        "kappa",
        "Ld",
        "ctrl_tgt_i",
        "lane_seg_count",
        "center_pt_count",
        "left_fit_ok",
        "right_fit_ok",
        "frame_idx",
        "camera_ok",
        "perception_ms",
        "control_ms",
        "loop_ms",
        "x",
        "y",
        "v",
    ]
    df = _to_numeric(df, numeric_cols)

    # Separate out runtime rows from reference seed rows
    if "source" in df.columns:
        runtime_df = df[df["source"] != "REF_SEED"].copy()
    else:
        runtime_df = df.copy()

    if len(runtime_df) == 0:
        print("No runtime rows found in CSV.")
        sys.exit(1)

    # Build relative time axis if timestamps exist
    if "t" in runtime_df.columns and runtime_df["t"].notna().any():
        t0 = runtime_df["t"].dropna().iloc[0]
        runtime_df["t_rel"] = runtime_df["t"] - t0
    else:
        runtime_df["t_rel"] = range(len(runtime_df))

    # Helpful console summary
    print(f"Runtime rows: {len(runtime_df)}")

    if "lane_conf" in runtime_df.columns:
        lc = runtime_df["lane_conf"].dropna()
        if len(lc) > 0:
            print(f"lane_conf min/max/mean = {lc.min():.3f} / {lc.max():.3f} / {lc.mean():.3f}")

    if "failsafe" in runtime_df.columns:
        fs = runtime_df["failsafe"].fillna(0)
        print(f"failsafe active rows = {int((fs > 0).sum())}/{len(fs)}")

    if "cmd_reason" in runtime_df.columns:
        print("\ncmd_reason counts:")
        print(runtime_df["cmd_reason"].fillna("").value_counts(dropna=False).head(10))

    if "source" in runtime_df.columns:
        print("\nsource counts:")
        print(runtime_df["source"].fillna("").value_counts(dropna=False))

    # 1) Perception confidence / failsafe / bad frames
    _plot_lines(
        runtime_df,
        "t_rel",
        ["lane_conf", "failsafe", "bad_frames"],
        "Perception Confidence / Failsafe",
        "value",
    )

    # 2) PWM outputs
    _plot_lines(
        runtime_df,
        "t_rel",
        ["steer_pwm_us", "throttle_pwm_us"],
        "Actuation PWM",
        "PWM (us)",
    )

    # 3) Command-level outputs
    _plot_lines(
        runtime_df,
        "t_rel",
        ["steer_rad", "accel_cmd"],
        "Controller Commands",
        "command value",
    )

    # 4) Controller debug terms
    _plot_lines(
        runtime_df,
        "t_rel",
        ["cte", "heading_err", "cte_term", "kappa", "Ld"],
        "Controller Debug Terms",
        "value",
    )

    # 5) Perception internals
    _plot_lines(
        runtime_df,
        "t_rel",
        ["lane_seg_count", "center_pt_count", "left_fit_ok", "right_fit_ok", "camera_ok"],
        "Perception Internals",
        "count / flag",
    )

    # 6) Timing diagnostics
    _plot_lines(
        runtime_df,
        "t_rel",
        ["perception_ms", "control_ms", "loop_ms"],
        "Timing Diagnostics",
        "milliseconds",
    )

    # 7) Optional vehicle state plot
    _plot_lines(
        runtime_df,
        "t_rel",
        ["v"],
        "Vehicle Speed",
        "m/s",
    )

    # 8) Optional frame index trend
    _plot_lines(
        runtime_df,
        "t_rel",
        ["frame_idx"],
        "Frame Index",
        "index",
    )

    plt.show()


if __name__ == "__main__":
    main()