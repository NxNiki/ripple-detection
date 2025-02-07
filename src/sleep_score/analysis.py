

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from debugpy.common.log import warning
from scipy.ndimage import gaussian_filter1d
from scipy.stats import stats
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

SECONDS_PER_HOUR = 3600
REM_CODE = -1
NON_REM_CODE = 1

SLEEP_STAGE_COLORS: Dict[int, str] = {REM_CODE: 'blue', NON_REM_CODE: 'green', 0: 'red'}
SLEEP_STAGE_LABELS: Dict[int, str] = {REM_CODE: 'REM', NON_REM_CODE: 'non-REM', 0: 'Other'}


def downsample_sleep_score(sleep_score: np.ndarray, original_sr: int = 2000, bin_size: int = 30) -> np.ndarray:
    """Downsample sleep_score_vec to bins of 30 seconds based on given rules."""

    sleep_score = sleep_score.squeeze()
    samples_per_bin = int(original_sr * bin_size)
    # num_bins = int(np.ceil(sleep_score_vec.shape[0] / samples_per_bin))
    num_bins = int(sleep_score.shape[0] // samples_per_bin)

    downsampled = np.zeros(num_bins, dtype=int)
    for i in range(num_bins):
        start_index = i * samples_per_bin
        end_index = min(start_index + samples_per_bin, sleep_score.shape[0])
        bin_values = sleep_score[start_index: end_index]
        unique, counts = np.unique(bin_values, return_counts=True)
        bin_dict = dict(zip(unique, counts))

        if REM_CODE in bin_dict and bin_dict[REM_CODE] > bin_dict.get(NON_REM_CODE, 0):
            downsampled[i] = REM_CODE
        elif NON_REM_CODE in bin_dict and bin_dict[NON_REM_CODE] > bin_dict.get(REM_CODE, 0):
            downsampled[i] = NON_REM_CODE

    return downsampled


def plot_sleep_stages(sleep_scores: np.ndarray) -> None:
    """
    Plots sleep stages based on a NumPy array of sleep scores.

    Args:
        sleep_scores (np.ndarray): A NumPy array of sleep scores where:
            -1.0 represents REM,
            1.0 represents non-REM,
            0.0 represents Other.
    """

    # Create a time axis (assuming each score corresponds to a 30-second epoch)
    time: np.ndarray = np.arange(len(sleep_scores)) * 30 / 60  # Convert to minutes

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 4))

    # Plot each sleep stage as a block (without edge color)
    for i, score in enumerate(sleep_scores):
        ax.barh(0, width=0.5, height=1, left=time[i], color=SLEEP_STAGE_COLORS[score])

    # Set labels and title
    ax.set_xlabel('Time (minutes)')
    ax.set_yticks([])  # Hide y-axis ticks
    ax.set_title('Sleep Stages')

    # Create legend
    legend_elements: list[plt.Rectangle] = [
        plt.Rectangle((0, 0), 1, 1, color=SLEEP_STAGE_COLORS[score], label=SLEEP_STAGE_LABELS[score])
        for score in SLEEP_STAGE_COLORS
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    # Show the plot
    plt.show()


def plot_two_sleep_stages(sleep_scores_1: np.ndarray, sleep_scores_2: np.ndarray, labels: List[str]) -> None:
    """
    Plots two sleep scores stacked vertically with labels.

    Args:
        sleep_scores_1 (np.ndarray): First sleep score array.
        sleep_scores_2 (np.ndarray): Second sleep score array.
        labels (List[str]): Labels for the two sleep scores (e.g., ['Subject 1', 'Subject 2']).
    """
    # Define the colors and labels for each sleep stage


    # Create a time axis (assuming each score corresponds to a 30-second epoch)
    time: np.ndarray = np.arange(len(sleep_scores_1)) * 30 / 60  # Convert to minutes

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 3))

    # Plot the first sleep score
    for i, score in enumerate(sleep_scores_1):
        ax.barh(0, width=0.5, height=1, left=time[i], color=SLEEP_STAGE_COLORS[score], edgecolor='none')

    # Plot the second sleep score
    for i, score in enumerate(sleep_scores_2):
        ax.barh(1, width=0.5, height=1, left=time[i], color=SLEEP_STAGE_COLORS[score], edgecolor='none')

    # Set labels and title
    ax.set_xlabel('Time (minutes)')
    ax.set_yticks([0, 1])  # Set y-ticks to correspond to the two sleep scores
    ax.set_yticklabels(labels)  # Add labels for the two sleep scores
    ax.set_title('Sleep Stages Comparison')

    # Create legend for sleep stages
    legend_elements: List[plt.Rectangle] = [
        plt.Rectangle((0, 0), 1, 1, color=SLEEP_STAGE_COLORS[score], label=SLEEP_STAGE_LABELS[score])
        for score in SLEEP_STAGE_COLORS
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    # Show the plot
    plt.tight_layout()
    plt.show()


def plot_contingency_matrix(
        sleep_scores_1: np.ndarray,
        sleep_scores_2: np.ndarray,
        method_labels: List[str]
) -> None:
    """
    Calculates and visualizes the contingency matrix of two sleep scores, ignoring elements with 0 ('Other').
    Adds precision, recall, and F1 score to the plot.

    Args:
        sleep_scores_1 (np.ndarray): First sleep score array.
        sleep_scores_2 (np.ndarray): Second sleep score array.
        method_labels (List[str]): Labels for the sleep score methods (e.g., ['Method A', 'Method B']).
    """
    # Filter out elements where either score is 0
    if len(sleep_scores_1) != len(sleep_scores_2):
        warning("unequal length of sleep scores!")
        min_length = min([len(sleep_scores_1), len(sleep_scores_2)])
        sleep_scores_1 = sleep_scores_1[:min_length]
        sleep_scores_2 = sleep_scores_2[:min_length]

    mask = (sleep_scores_1 != 0) & (sleep_scores_2 != 0)
    filtered_scores_1 = sleep_scores_1[mask]
    filtered_scores_2 = sleep_scores_2[mask]

    # Calculate the contingency matrix
    value_labels = ['REM', 'non-REM']  # Labels for -1 and 1
    cm = confusion_matrix(filtered_scores_1, filtered_scores_2, labels=[-1, 1])

    # Calculate metrics
    precision = precision_score(filtered_scores_1, filtered_scores_2, labels=[-1, 1], average='weighted')
    recall = recall_score(filtered_scores_1, filtered_scores_2, labels=[-1, 1], average='weighted')
    f1 = f1_score(filtered_scores_1, filtered_scores_2, labels=[-1, 1], average='weighted')

    # Create a heatmap for visualization
    plt.figure(figsize=(5, 3))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=value_labels,
        yticklabels=value_labels
    )
    plt.xlabel(method_labels[1])  # Label for sleep_scores_2
    plt.ylabel(method_labels[0])  # Label for sleep_scores_1
    plt.title('Contingency Matrix (Ignoring "Other")', fontsize=10)

    # Add metrics to the plot
    metrics_text = f'Precision: {precision:.2f}\nRecall: {recall:.2f}\nF1 Score: {f1:.2f}'
    plt.text(
        x=1.3,
        y=0.7,
        s=metrics_text,
        ha='left',
        va='bottom',
        transform=plt.gca().transAxes,
        bbox=dict(facecolor='white', alpha=0.8)
    )

    plt.tight_layout()
    plt.show()


def load_sleep_score(file_name: str) -> np.ndarray:
    """
    Reads the first column of a .csv file, ignores the header, and maps values:
    - 'Wake/REM' to -1
    - All other values to 1.

    Args:
        file_path (str): Path to the .csv file.

    Returns:
        np.ndarray: A NumPy array of integer values.
    """

    # Read the first column of the .csv file, skipping the header
    df = pd.read_csv(file_name, usecols=[0], header=None, skiprows=1)

    # Map 'Wake/REM' to -1 and all other values to 1
    mapped_values = df[0].apply(lambda x: REM_CODE if x.strip() == 'Wake/REM' else NON_REM_CODE)

    # Convert the mapped values to a NumPy array of integers
    return mapped_values.to_numpy()


def plot_ripple_rate_by_sleep_state(
        ripple_indices: np.ndarray,
        sleep_states: np.ndarray,
        sampling_rate: int = 500,
        sleep_score_sr: float = 1 / 30,
        segment_duration: float = 1,
        event_name: str = 'Ripple Rate',
):
    """
    Match ripple indices to sleep states and compute ripple rates for REM and non-REM sleep.

    Parameters:
        ripple_indices (np.ndarray): Indices of ripple start times.
        sleep_states (np.ndarray): Array with -1 (non-REM), 0 (unknown), and 1 (REM).
        sampling_rate (int): Sampling rate of the ripple_indices in Hz (default is 500 Hz).
        sleep_score_sr (float): Sampling rate of sleep_states in Hz (default is 1/30 Hz).
        segment_duration (float): split plots for each segment (default is 1 hour).

    Returns:
        None: Displays a bar plot of ripple rates for REM and non-REM sleep.
    """

    ripple_times_sec = ripple_indices/ sampling_rate
    sleep_state_times = np.arange(len(sleep_states)) / sleep_score_sr
    sleep_state_for_ripples = np.digitize(ripple_times_sec, sleep_state_times)

    # Remove ripples that fall outside the sleep state range
    valid_mask = (sleep_state_for_ripples >= 0) & (sleep_state_for_ripples < len(sleep_states))
    ripple_bins = sleep_state_for_ripples[valid_mask]

    # Count ripples per 30s bin
    unique_bins, ripple_counts = np.unique(ripple_bins, return_counts=True)
    ripple_rates = np.zeros_like(sleep_states, dtype=float)
    ripple_rates[unique_bins] = ripple_counts  # Assign counts to correct bins

    # Create DataFrame for plotting
    df = pd.DataFrame({"Ripple Rate": ripple_rates, "Sleep State": sleep_states})
    df["Segment"] = np.floor(np.arange(len(df)) / sleep_score_sr / SECONDS_PER_HOUR / segment_duration).astype(int)
    df = df[df["Sleep State"] != 0]  # Remove unknown states

    # Replace -1 and 1 with labels
    df["Sleep State"] = df["Sleep State"].replace({NON_REM_CODE: "Non-REM", REM_CODE: "REM"})

    # Plot box plots
    plt.figure(figsize=(6, 4))
    sns.boxplot(x="Segment",
                y="Ripple Rate",
                hue="Sleep State",
                data=df,
                palette={"Non-REM": SLEEP_STAGE_COLORS[NON_REM_CODE], "REM": SLEEP_STAGE_COLORS[REM_CODE]})
    plt.ylabel(f"{event_name} (per {1/sleep_score_sr}s)")
    plt.title(f"{event_name} in REM vs Non-REM Sleep")
    plt.show()


def plot_activation_by_ripple(
        activation: np.ndarray,
        ripple_start_index: np.ndarray,
        ripple_end_index: np.ndarray,
        activation_fs: int,
        ripple_fs: int,
        activation_label: str ,
        event_label: str = 'Ripple',
) -> None:
    """
    Plot model activation around ripple index vs others.

    Parameters:
        activation (np.ndarray): Activation signal.
        ripple_start_index (np.ndarray): Indices where ripples start.
        ripple_end_index (np.ndarray): Indices where ripples end.
        activation_fs (int): Sampling rate of activation.
        ripple_fs (int): Sampling rate of ripple events.
        activation_label (str): Label for activation type.
        event_label (str): Label for event (ripple, spindle, etc.).
    """

    activation = activation.squeeze()
    activation_time = np.arange(len(activation)) / activation_fs
    ripple_start_time = ripple_start_index / ripple_fs
    ripple_end_time = ripple_end_index / ripple_fs
    activation_around_ripple_index = np.zeros(len(activation), dtype=bool)

    ripple_activation = []
    for start_index, end_index in get_activation_index_around_ripple(activation_time, ripple_start_time,
                                                                     ripple_end_time):
        activation_around_ripple_index[start_index: end_index+1] = 1
        ripple_activation.append(np.mean(activation[start_index: end_index+1]))

    ripple_activation = np.array(ripple_activation)
    non_ripple_activation = np.mean(activation[activation_around_ripple_index == 0])

    # Smoothed activation for visualization (does not affect mean calculation)
    smoothed_activation = gaussian_filter1d(activation, sigma=100)

    # Create subplots (side-by-side)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={'width_ratios': [1, 4]})

    # Run t-test
    t_stat, p_value = stats.ttest_1samp(ripple_activation, non_ripple_activation)
    plot_ripple_activation = np.random.choice(ripple_activation, 1000, replace=False)

    # Box plot (Ripple Activation)
    # axes[0].boxplot(ripple_activation, positions=[1], widths=0.6, patch_artist=True,
    #                 boxprops=dict(facecolor='blue', alpha=0.5), medianprops=dict(color='black'), flierprops=dict(marker=''))
    sns.swarmplot(x=np.ones_like(plot_ripple_activation), y=plot_ripple_activation, color='black', size=1, ax=axes[0])
    axes[0].axhline(non_ripple_activation, color='red', linestyle='--', linewidth=1, label=f'Non-{event_label}')
    axes[0].axhline(np.mean(ripple_activation), color='black', linestyle='--', linewidth=1, label=f'{event_label}')

    # Format box plot
    # axes[0].set_xticks([1])
    axes[0].set_xticklabels([event_label])
    axes[0].set_ylabel(f'Mean {activation_label} for {event_label}')
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    axes[0].legend(loc="best", fontsize=7)

    # Display t-test result
    axes[0].text(0, 1.05, f'p = {p_value:.3f}', ha='center', fontsize=10)

    # Activation curve plot with shaded ripple regions
    axes[1].plot(activation_time / SECONDS_PER_HOUR, smoothed_activation, color='blue', label=f'{activation_label}', alpha=.7)
    # for start, end in zip(ripple_start_time / SECONDS_PER_HOUR, ripple_end_time / SECONDS_PER_HOUR):
    #     # axes[1].axvspan(start, end, color='blue', alpha=0.2)  # Shade ripple intervals
    #     # axes[1].axvline(start, color='blue', linestyle='--', alpha=0.2)
    axes[1].scatter(ripple_start_time / SECONDS_PER_HOUR, np.random.rand(len(ripple_start_time)), s=.1, color='k')  # SWR time v. trial

    axes[1].set_xlabel('Time (h)')
    axes[1].set_ylabel(f'{activation_label}')
    axes[1].set_title(f'{activation_label} Over Time with {event_label} Intervals')
    #axes[1].legend()
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()


def get_activation_index_around_ripple(
        activation_time: np.ndarray,
        ripple_start_time: np.ndarray,
        ripple_end_time: np.ndarray,
) -> Tuple[int, int]:

    if len(ripple_start_time) != len(ripple_end_time):
        raise ValueError("start and end time should have equal length!")

    for i in range(len(ripple_start_time)):
        start_time = ripple_start_time[i]
        end_time = ripple_end_time[i]
        start_index = np.searchsorted(activation_time, start_time, side='right')
        end_index = np.searchsorted(activation_time, end_time, side='left')
        end_index = min([end_index, len(activation_time)-1])
        yield start_index, end_index
