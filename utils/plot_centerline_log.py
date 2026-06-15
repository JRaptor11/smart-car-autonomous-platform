import sys

import pandas as pd
import matplotlib.pyplot as plt

from typing import List

def _to_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_centerline_log.py <path/to/centerline_log.csv>")
        sys.exit(1)

    csv_path = sys.argv[1]
    df = pd.read_csv(csv_path)

    df = _to_numeric(
        df,
        ["t", "frame_idx", "pt_idx", "x_local_m", "y_local_m", "lane_conf"],
    )

    df = df.dropna(subset=["x_local_m", "y_local_m"])
    if len(df) == 0:
        print("No valid centerline points found.")
        sys.exit(1)

    # Build relative time if available
    if "t" in df.columns and df["t"].notna().any():
        t0 = df["t"].dropna().iloc[0]
        df["t_rel"] = df["t"] - t0
    else:
        df["t_rel"] = 0.0

    print(f"Rows: {len(df)}")
    if "frame_idx" in df.columns and df["frame_idx"].notna().any():
        print(f"Frames: {df['frame_idx'].nunique()}")

    # ----- Plot 1: overlay all centerlines -----
    plt.figure(figsize=(8, 6))
    for frame_id, g in df.groupby("frame_idx", dropna=False):
        g = g.sort_values("pt_idx")
        plt.plot(
            g["x_local_m"],
            g["y_local_m"],
            alpha=0.15,
            linewidth=1.0,
            color="tab:blue",
        )

    plt.axhline(0.0, linestyle="--", linewidth=1.0, color="black", alpha=0.6, label="Vehicle center")
    plt.xlabel("Forward distance x_local (m)")
    plt.ylabel("Lateral offset y_local (m)")
    plt.title("Detected Centerlines (all frames overlay)")
    plt.grid(True)
    plt.legend()

    # ----- Plot 2: time-colored nearest point per frame -----
    nearest_rows = []
    for frame_id, g in df.groupby("frame_idx", dropna=False):
        g = g.copy()
        g["dist2"] = g["x_local_m"] ** 2 + g["y_local_m"] ** 2
        nearest_rows.append(g.loc[g["dist2"].idxmin()])

    frame_nearest = pd.DataFrame(nearest_rows)

    plt.figure(figsize=(8, 6))
    sc = plt.scatter(
        frame_nearest["x_local_m"],
        frame_nearest["y_local_m"],
        c=frame_nearest["t_rel"] if "t_rel" in frame_nearest.columns else range(len(frame_nearest)),
        cmap="viridis",
        s=30,
        alpha=0.9,
        label="Nearest centerline point per frame",
    )
    plt.axhline(0.0, linestyle="--", linewidth=1.0, color="black", alpha=0.6, label="Vehicle center")
    plt.xlabel("Forward distance x_local (m)")
    plt.ylabel("Lateral offset y_local (m)")
    plt.title("Nearest Detected Centerline Point Over Time")
    plt.grid(True)
    plt.legend()
    cbar = plt.colorbar(sc)
    cbar.set_label("Time (relative s)")

    # ----- Plot 3: single best-confidence frame -----
    if "lane_conf" in df.columns and df["lane_conf"].notna().any():
        frame_conf = df.groupby("frame_idx", as_index=False)["lane_conf"].mean()
        best_frame = frame_conf.sort_values("lane_conf", ascending=False).iloc[0]["frame_idx"]
        best = df[df["frame_idx"] == best_frame].sort_values("pt_idx")

        plt.figure(figsize=(8, 6))
        plt.plot(
            best["x_local_m"],
            best["y_local_m"],
            linewidth=2.0,
            color="tab:red",
            label=f"Selected high-confidence frame {int(best_frame) if pd.notna(best_frame) else best_frame}",
        )
        plt.axhline(0.0, linestyle="--", linewidth=1.0, color="black", alpha=0.6, label="Vehicle center")
        plt.xlabel("Forward distance x_local (m)")
        plt.ylabel("Lateral offset y_local (m)")
        plt.title("Single Detected Centerline (selected high-confidence frame)")
        plt.grid(True)
        plt.legend()

    plt.show()


if __name__ == "__main__":
    main()