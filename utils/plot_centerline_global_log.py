import sys

import pandas as pd
import matplotlib.pyplot as plt


def _to_numeric(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_centerline_global_log.py <path/to/centerline_global_log.csv>")
        sys.exit(1)

    csv_path = sys.argv[1]
    df = pd.read_csv(csv_path)

    df = _to_numeric(
        df,
        [
            "t",
            "frame_idx",
            "pt_idx",
            "x_local_m",
            "y_local_m",
            "x_world_m",
            "y_world_m",
            "vehicle_x",
            "vehicle_y",
            "vehicle_yaw",
            "lane_conf",
        ],
    )

    df = df.dropna(subset=["x_world_m", "y_world_m"])
    if len(df) == 0:
        print("No valid global centerline points found.")
        sys.exit(1)

    if "t" in df.columns and df["t"].notna().any():
        t0 = df["t"].dropna().iloc[0]
        df["t_rel"] = df["t"] - t0
    else:
        df["t_rel"] = 0.0

    print(f"Rows: {len(df)}")
    if "frame_idx" in df.columns and df["frame_idx"].notna().any():
        print(f"Frames: {df['frame_idx'].nunique()}")

    # Plot 1: all global centerlines overlay
    plt.figure(figsize=(8, 6))
    for frame_id, g in df.groupby("frame_idx", dropna=False):
        g = g.sort_values("pt_idx")
        plt.plot(
            g["x_world_m"],
            g["y_world_m"],
            alpha=0.15,
            linewidth=1.0,
            color="tab:blue",
        )

    # Overlay estimated vehicle path from logged poses
    vehicle_pts = (
        df.sort_values(["frame_idx", "pt_idx"])
          .groupby("frame_idx", as_index=False)
          .first()
    )

    plt.plot(
        vehicle_pts["vehicle_x"],
        vehicle_pts["vehicle_y"],
        linewidth=2.0,
        color="tab:red",
        label="Estimated vehicle path",
    )

    plt.xlabel("World X (m)")
    plt.ylabel("World Y (m)")
    plt.title("Reconstructed Global Centerlines")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()

    # Plot 2: best-confidence single frame in world space
    if "lane_conf" in df.columns and df["lane_conf"].notna().any():
        frame_conf = df.groupby("frame_idx", as_index=False)["lane_conf"].mean()
        best_frame = frame_conf.sort_values("lane_conf", ascending=False).iloc[0]["frame_idx"]
        best = df[df["frame_idx"] == best_frame].sort_values("pt_idx")

        plt.figure(figsize=(8, 6))
        plt.plot(
            best["x_world_m"],
            best["y_world_m"],
            linewidth=2.0,
            color="tab:green",
            label=f"Best-confidence frame {int(best_frame) if pd.notna(best_frame) else best_frame}",
        )
        plt.scatter(
            [best["vehicle_x"].iloc[0]],
            [best["vehicle_y"].iloc[0]],
            color="tab:red",
            s=40,
            label="Vehicle pose for that frame",
        )
        plt.xlabel("World X (m)")
        plt.ylabel("World Y (m)")
        plt.title("Single Reconstructed Centerline in World Space")
        plt.axis("equal")
        plt.grid(True)
        plt.legend()

    plt.show()


if __name__ == "__main__":
    main()