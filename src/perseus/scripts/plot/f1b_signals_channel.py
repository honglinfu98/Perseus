from os import path
import matplotlib.pyplot as plt
import seaborn as sns
from perseus.dataset.extract.cloudburst_connection import get_signals_channel
from perseus.settings import PROJECT_ROOT


def plot_combined_kde(crowd_pump_signals, time_pump_signals, font_size=45, bw=0.5):
    """
    Plots a combined KDE (Kernel Density Estimate) plot for Crowd Pump and Time Pump signals on a log-log scale.

    Parameters:
        crowd_pump_signals (array-like): Data for the Crowd Pump signals.
        time_pump_signals (array-like): Data for the Time Pump signals.
        font_size (int): Font size for all text elements in the plot.
        bw (float): Bandwidth adjustment factor for the KDE plot.
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(9, 6.5))

    # Plot KDE for Crowd Pump with filled area
    sns.kdeplot(
        crowd_pump_signals,
        bw_adjust=bw,
        log_scale=True,
        label="Crowd Pump",
        color="C0",
        fill=True,
        alpha=0.5,  # Adjust transparency for better visualization
    )

    # Plot KDE for Time Pump with filled area
    sns.kdeplot(
        time_pump_signals,
        bw_adjust=bw,
        log_scale=True,
        label="Time Pump",
        color="C1",
        fill=True,
        alpha=0.5,  # Adjust transparency for better visualization
    )

    # Set both axes to logarithmic scale
    ax.set_xscale("log")
    ax.set_yscale("log")

    # Set the axis labels and title with the specified font size
    ax.set_xlabel("Number of Signals", fontsize=font_size)
    ax.set_ylabel("Density", fontsize=font_size)

    # Set tick parameters to use the specified font size
    ax.tick_params(axis="both", which="major", labelsize=font_size)
    ax.tick_params(axis="both", which="minor", labelsize=font_size)

    # Set legend font size
    ax.legend(prop={"size": font_size}, loc="lower left")
    plt.tight_layout()

    plt.savefig(path.join(PROJECT_ROOT, "data", "feb_signals_per_channel_kde.pdf"))
    plt.show()


# Get signals data
df = get_signals_channel()

crowd_pump_signals = df["crowd_pump_count"][df["crowd_pump_count"] > 0].values
time_pump_signals = df["time_pump_count"][df["time_pump_count"] > 0].values

# Example usage:
plot_combined_kde(crowd_pump_signals, time_pump_signals, font_size=35, bw=0.8)
