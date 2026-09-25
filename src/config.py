"""Define configuration settings and filesystem paths for the analysis pipeline."""

from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
DATA_OUT_DIR = OUTPUT_DIR / "data"
PLOTS_DIR = OUTPUT_DIR / "plots"

RAW_DATA_FILE = DATA_DIR / "pitch_serology_data.csv"
INVENTORY_FILE = DATA_OUT_DIR / "data_inventory.txt"
SUMMARY_FILE = OUTPUT_DIR / "statistical_summary.txt"

RANDOM_STATE = 42
N_PERMUTATIONS = 1000
CV_FOLDS = 5


def make_output_dirs() -> None:
    """Create output directories if they do not exist."""
    for directory in [OUTPUT_DIR, CHECKPOINT_DIR, DATA_OUT_DIR, PLOTS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def setup_plotting_theme() -> None:
    """Configure plot style parameters for publication figures."""
    plt.rcParams.update({
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'figure.figsize': (6.5, 5),
        'font.size': 11,
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'legend.title_fontsize': 11,
        'lines.linewidth': 2.0,
        'lines.markersize': 6,
    })
    sns.set_theme(style="ticks", context="paper")


PALETTE = sns.color_palette("colorblind")
