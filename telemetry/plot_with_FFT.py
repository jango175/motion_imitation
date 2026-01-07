import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import pi
from statistics import mean, stdev


class IMUTelemetryAnalyzer:
    def __init__(self, filename):
        self.filename = filename
        self.data = pd.read_csv(filename)

        # ensure sorted by time
        self.data = self.data.sort_values("timestamp").reset_index(drop=True)

        # relative time
        self.data["time"] = self.data["timestamp"] - self.data["timestamp"].iloc[0]

        # latex formatting
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "monospace",
            "font.monospace": 'Computer Modern Typewriter'
        })

    # -------------------------------------------------
    # Data cutting
    # -------------------------------------------------
    def cut_start(self, start_time_sec, stop_time_sec):
        self.data = self.data[(self.data["time"] >= start_time_sec) & (self.data["time"] <= stop_time_sec)].reset_index(drop=True)
        self.data["time"] -= self.data["time"].iloc[0]
        print(f"Cut data before {start_time_sec:.2f}s -> {len(self.data)} samples remain")

    # -------------------------------------------------
    # Pretty statistics
    # -------------------------------------------------
    def print_stats(self):
        print("\n=== BASIC STATISTICS ===")

        for name in ["gyro_x", "gyro_y", "gyro_z"]:
            vals = self.data[name]
            print(
                f"{name:8s} | mean={mean(vals): .4f} rad/s "
                f"std={stdev(vals): .4f} "
                f"min={min(vals): .4f} "
                f"max={max(vals): .4f}"
            )

        for name in ["accel_x", "accel_y", "accel_z"]:
            vals = self.data[name]
            print(
                f"{name:8s} | mean={mean(vals): .4f} m/s^2 "
                f"std={stdev(vals): .4f} "
                f"min={min(vals): .4f} "
                f"max={max(vals): .4f}"
            )

    # -------------------------------------------------
    # Time-domain plots
    # -------------------------------------------------
    def plot_time(self):
        t = self.data["time"]

        plt.figure(figsize=(10, 6))
        plt.plot(t.to_numpy(), self.data["gyro_x"].to_numpy(), label="Gyro X")
        plt.plot(t.to_numpy(), self.data["gyro_y"].to_numpy(), label="Gyro Y")
        plt.plot(t.to_numpy(), self.data["gyro_z"].to_numpy(), label="Gyro Z")
        plt.xlabel("Time [s]")
        plt.ylabel("Angular Velocity [$\\frac{^\circ}{s}$]")
        plt.title("\\bf{Gyroscope data}")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.plot(t.to_numpy(), self.data["accel_x"].to_numpy(), label="Accel X")
        plt.plot(t.to_numpy(), self.data["accel_y"].to_numpy(), label="Accel Y")
        plt.plot(t.to_numpy(), self.data["accel_z"].to_numpy(), label="Accel Z")
        plt.xlabel("Time [s]")
        plt.ylabel("Acceleration [$\\frac{m}{s^2}$]")
        plt.title("\\bf{Accelerometer data}")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()

    # -------------------------------------------------
    # FFT utility
    # -------------------------------------------------
    def _fft(self, signal, fs):
        n = len(signal)
        signal = signal - np.mean(signal)  # remove DC
        fft_vals = np.fft.rfft(signal)
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        mag = np.abs(fft_vals) / n
        return freqs, mag

    # -------------------------------------------------
    # FFT plots
    # -------------------------------------------------
    def plot_fft(self):
        t = self.data["time"].values
        dt = np.mean(np.diff(t))
        fs = 1.0 / dt

        print(f"\nEstimated sampling frequency: {fs:.1f} Hz")

        plt.figure(figsize=(10, 6))
        for axis in ["gyro_x", "gyro_y", "gyro_z"]:
            f, mag = self._fft(self.data[axis].values, fs)
            # change first letter to capital for legend
            label = axis.capitalize()
            plt.plot(f, mag, label=label)

        plt.xlim(0, 40)  # walking dynamics range
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Magnitude [$\\frac{^\circ}{s}$]")
        plt.title("\\bf{Gyroscope FFT data}")
        plt.legend()
        plt.grid()
        # plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10, 6))
        for axis in ["accel_x", "accel_y", "accel_z"]:
            f, mag = self._fft(self.data[axis].values, fs)
            label = axis.capitalize()
            plt.plot(f, mag, label=label)

        plt.xlim(0, 40)
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Magnitude [$\\frac{m}{s^2}$]")
        plt.title("\\bf{Accelerometer FFT data}")
        plt.legend()
        plt.grid()
        # plt.tight_layout()
        plt.show()


# -------------------------------------------------
# Example usage
# -------------------------------------------------
if __name__ == "__main__":
    # analyzer = IMUTelemetryAnalyzer("log_01_walki_PID.csv")
    analyzer = IMUTelemetryAnalyzer('telemetry_log_20251214_235450.csv')

    analyzer.cut_start(3.0, 12.0)
    analyzer.print_stats()
    analyzer.plot_time()
    analyzer.plot_fft()
