import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


File_1 = r"E:\UCSD\Winter2026\ECE257B\Project\data_reading\data\low gain+high gain\test_2ant_btftbF_2026-02-25_17-47-06.csv"
C = 3e8

# =========================================================
# CREATE OUTPUT FOLDER FOR DATASET
# =========================================================
SAVE_DIR = "low+high"

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

df = pd.read_csv(
    File_1,
    comment='/',
    header=None
)

df.columns = [
    "Timestamp",
    "EPC",
    "TID",
    "Antenna",
    "RSSI",
    "Frequency",
    "Hostname",
    "PhaseAngle",
    "DopplerFrequency",
    "CRHandle"
]

# Count reads per tag
tid_counts = df["TID"].value_counts()

# Get the 2 most read TIDs
top2_tids = tid_counts.head(2).index

print("Top 2 TIDs:", top2_tids.tolist())

# Filter dataframe
df = df[df["TID"].isin(top2_tids)].copy()

# Optional: reset index after filtering
df.reset_index(drop=True, inplace=True)

df["Timestamp"] = pd.to_datetime(df["Timestamp"])

df["TimeSeconds"] = (
    df["Timestamp"] - df["Timestamp"].iloc[0]
).dt.total_seconds()

antennas = df["Antenna"].unique()

sample_counter = 0

for ant in antennas:

    print(f"\nProcessing Antenna {ant}")

    ant_df = df[df["Antenna"] == ant].copy()

    ant_df = ant_df.sort_values("TimeSeconds")

    phase = ant_df["PhaseAngle"].values

    time = ant_df["TimeSeconds"].values

    freq = ant_df["Frequency"].iloc[0] * 1e6  # MHz → Hz

    LAMBDA = C / freq

    # =========================================================
    # PHASE UNWRAPPING
    # =========================================================

    unwrapped = [phase[0]]

    for i in range(1, len(phase)):

        delta = phase[i] - phase[i-1]

        if delta > np.pi:
            delta -= 2*np.pi

        elif delta < -np.pi:
            delta += 2*np.pi

        unwrapped.append(unwrapped[-1] + delta)

    unwrapped = np.array(unwrapped)

    # =========================================================
    # SMOOTH SIGNAL
    # =========================================================

    N = int(len(unwrapped)/5)

    unwrapped = np.convolve(
        unwrapped,
        np.ones(N)/N,
        mode='valid'
    )

    time = time[0:-N+1]

    # =========================================================
    # ADD SYNTHETIC RF NOISE
    # =========================================================

    # Gaussian noise
    gaussian_noise = np.random.normal(
        0,
        1.5,
        len(unwrapped)
    )

    # Multipath-like oscillatory distortion
    multipath_noise = (
        1.2 * np.sin(2 * np.pi * 0.7 * time)
        +
        0.8 * np.sin(2 * np.pi * 2.3 * time)
    )

    # Sudden RFID phase jumps
    jump_noise = np.zeros(len(unwrapped))

    jump_indices = np.random.choice(
        len(unwrapped),
        size=10,
        replace=False
    )

    for idx in jump_indices:

        jump_noise[idx:] += np.random.uniform(-3, 3)

    # Burst interference
    burst_noise = np.zeros(len(unwrapped))

    for _ in range(6):

        start = np.random.randint(
            0,
            len(unwrapped) - 100
        )

        burst_noise[start:start+80] += np.random.normal(
            0,
            2.0,
            80
        )

    # =========================================================
    # FINAL NOISY SIGNAL
    # =========================================================

    noisy_unwrapped = (
        unwrapped
        + gaussian_noise
        + multipath_noise
        + jump_noise
        + burst_noise
    )

    # =========================================================
    # SAVE WINDOWS AS CSV FILES
    # =========================================================

    WINDOW = 256
    STRIDE = 256

    for i in range(0, len(unwrapped) - WINDOW, STRIDE):

        clean_window = unwrapped[i:i+WINDOW]

        noisy_window = noisy_unwrapped[i:i+WINDOW]

        time_window = time[i:i+WINDOW]

        sample_df = pd.DataFrame({
            "time": time_window,
            "clean": clean_window,
            "noisy": noisy_window
        })

        save_path = os.path.join(
            SAVE_DIR,
            f"sample_{sample_counter}.csv"
        )

        sample_df.to_csv(
            save_path,
            index=False
        )

        sample_counter += 1

    print(f"Saved {sample_counter} total training samples")

    # =========================================================
    # COMPUTE VELOCITY
    # =========================================================

    STEP = 100

    dt = time[STEP:] - time[:-STEP]

    dphi = (
        noisy_unwrapped[STEP:]
        -
        noisy_unwrapped[:-STEP]
    )

    phase_slope = dphi / dt

    velocity = (
        LAMBDA / (4 * np.pi)
    ) * phase_slope

    time_mid = time[STEP:]

    # =========================================================
    # PLOTS
    # =========================================================

    plt.figure(figsize=(12,8))

    plt.subplot(4,1,1)

    plt.plot(time, unwrapped)

    plt.title(
        f"Antenna {ant} - Clean Unwrapped Phase"
    )

    plt.ylabel("Radians")

    plt.grid()

    plt.subplot(4,1,2)

    plt.plot(time, noisy_unwrapped)

    plt.title("Very Noisy RF Signal")

    plt.ylabel("Radians")

    plt.grid()

    plt.subplot(4,1,3)

    plt.plot(time_mid, phase_slope)

    plt.title("Phase Slope (rad/sec)")

    plt.ylabel("rad/sec")

    plt.grid()

    plt.subplot(4,1,4)

    plt.plot(time_mid, velocity)

    plt.title("Velocity (m/s)")

    plt.ylabel("m/s")

    plt.xlabel("Time (s)")

    plt.grid()

    plt.tight_layout()

    plt.show()

print("\nDataset generation complete!")

print(f"Saved dataset to folder: {SAVE_DIR}")