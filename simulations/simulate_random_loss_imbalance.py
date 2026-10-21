"""Monte Carlo routing performance for random arm-loss imbalance in a 3x3 DFT MMI.

Each arm receives an independent Gaussian power-loss error in dB.  The common
component is removed per realization because it changes insertion loss but not
normalized routing performance.  Reflector losses are fixed and independent
of phase; phases are chosen optimally for each target output.
"""

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from tfln_paths import OUTPUT_ROOT


import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = OUTPUT_ROOT / "simulations"
OUTPUT_PNG = OUTPUT_DIR / "random_arm_loss_performance_3x3.png"
OUTPUT_SVG = OUTPUT_DIR / "random_arm_loss_performance_3x3.svg"
OUTPUT_CSV = OUTPUT_DIR / "random_arm_loss_performance_3x3.csv"
OUTPUT_PDF = OUTPUT_ROOT / "figures/supplementary" / "FIG_Random_Arm_Loss_3x3.pdf"
SEED = 7
DEVICES = 300_000

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "svg.fonttype": "none",
    }
)

BLUE = "#3378A8"
ORANGE = "#D97722"
GRAY = "#777777"


def optimal_metrics(
    loss_error_db: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return normalized target fraction, extinction ratio, and routing penalty."""
    power_returns = 10 ** (-loss_error_db / 10)
    amplitudes = np.sqrt(power_returns)
    target_fraction = amplitudes.sum(axis=1) ** 2 / (3 * power_returns.sum(axis=1))
    total_leakage = np.maximum(1 - target_fraction, np.finfo(float).tiny)
    extinction_db = 10 * np.log10(2 * target_fraction / total_leakage)
    routing_penalty_db = -10 * np.log10(target_fraction)
    return target_fraction, extinction_db, routing_penalty_db


def main() -> None:
    rng = np.random.default_rng(SEED)
    independent_errors = rng.normal(size=(DEVICES, 3))
    relative_errors = independent_errors - independent_errors.mean(
        axis=1, keepdims=True
    )
    sigma_grid = np.linspace(0.02, 3.0, 120)

    target_q = np.empty((len(sigma_grid), 3))
    extinction_q = np.empty((len(sigma_grid), 3))
    penalty_q = np.empty((len(sigma_grid), 3))
    yield_20 = np.empty(len(sigma_grid))
    yield_30 = np.empty(len(sigma_grid))
    yield_40 = np.empty(len(sigma_grid))

    for index, sigma_db in enumerate(sigma_grid):
        target, extinction, penalty = optimal_metrics(sigma_db * relative_errors)
        target_q[index] = np.percentile(target, [5, 50, 95])
        extinction_q[index] = np.percentile(extinction, [5, 50, 95])
        penalty_q[index] = np.percentile(penalty, [5, 50, 95])
        yield_20[index] = np.mean(extinction >= 20)
        yield_30[index] = np.mean(extinction >= 30)
        yield_40[index] = np.mean(extinction >= 40)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table = np.column_stack(
        (
            sigma_grid,
            100 * target_q,
            extinction_q,
            penalty_q,
            yield_20,
            yield_30,
            yield_40,
        )
    )
    np.savetxt(
        OUTPUT_CSV,
        table,
        delimiter=",",
        header=(
            "sigma_arm_loss_db,target_fraction_p05_percent,target_fraction_median_percent,"
            "target_fraction_p95_percent,extinction_p05_db,extinction_median_db,"
            "extinction_p95_db,routing_penalty_p05_db,routing_penalty_median_db,"
            "routing_penalty_p95_db,yield_er_ge_20,yield_er_ge_30,yield_er_ge_40"
        ),
        comments="",
    )

    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.65), constrained_layout=False)
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.21, top=0.76, wspace=0.38)
    fig.suptitle(
        "Ideal $3\\times3$ router with independent Gaussian arm-loss errors",
        fontsize=10,
        y=0.95,
    )

    ax = axes[0]
    ax.fill_between(
        sigma_grid,
        100 * target_q[:, 0],
        100 * target_q[:, 2],
        color=BLUE,
        alpha=0.20,
        label="5th–95th percentile",
    )
    ax.plot(sigma_grid, 100 * target_q[:, 1], color=BLUE, lw=1.8, label="median")
    ax.set_title("a) Best routed fraction", loc="left")
    ax.set_xlabel(r"Per-arm loss variation $\sigma$ (dB)")
    ax.set_ylabel("Target power / returned power (\%)")
    ax.set_xlim(0, 3)
    ax.set_ylim(78, 100.4)
    ax.grid(color="#DDDDDD", lw=0.5)
    ax.legend(loc="lower left", frameon=False)

    ax = axes[1]
    ax.fill_between(
        sigma_grid,
        extinction_q[:, 0],
        extinction_q[:, 2],
        color=BLUE,
        alpha=0.20,
    )
    ax.plot(sigma_grid, extinction_q[:, 1], color=BLUE, lw=1.8)
    ax.axhline(20, color=GRAY, ls="--", lw=0.8)
    ax.axhline(30, color=GRAY, ls=":", lw=0.8)
    ax.set_title("b) Optimized extinction", loc="left")
    ax.set_xlabel(r"Per-arm loss variation $\sigma$ (dB)")
    ax.set_ylabel("Worst-case extinction ratio (dB)")
    ax.set_xlim(0, 3)
    ax.set_ylim(5, 60)
    ax.grid(color="#DDDDDD", lw=0.5)

    ax = axes[2]
    ax.plot(sigma_grid, 100 * yield_20, color=BLUE, lw=1.8, label=r"ER $\geq20$ dB")
    ax.plot(sigma_grid, 100 * yield_30, color=ORANGE, lw=1.6, label=r"ER $\geq30$ dB")
    ax.plot(sigma_grid, 100 * yield_40, color=GRAY, lw=1.3, label=r"ER $\geq40$ dB")
    ax.set_title("c) Device yield", loc="left")
    ax.set_xlabel(r"Per-arm loss variation $\sigma$ (dB)")
    ax.set_ylabel("Fraction of devices (\%)")
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 102)
    ax.grid(color="#DDDDDD", lw=0.5)
    ax.legend(loc="upper right", frameon=False)

    fig.savefig(OUTPUT_PNG, dpi=300)
    fig.savefig(OUTPUT_SVG)
    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PDF)
    plt.close(fig)

    for sigma_db in (0.25, 0.5, 1.0, 2.0):
        target, extinction, penalty = optimal_metrics(sigma_db * relative_errors)
        print(
            f"sigma={sigma_db:.2f} dB: "
            f"target p05/median/p95={np.percentile(100 * target, [5, 50, 95])}, "
            f"ER p05/median/p95={np.percentile(extinction, [5, 50, 95])}, "
            f"target penalty median={np.median(penalty):.4f} dB, "
            f"yield ER>=20/30/40={np.mean(extinction >= 20):.4f}/"
            f"{np.mean(extinction >= 30):.4f}/{np.mean(extinction >= 40):.4f}"
        )


if __name__ == "__main__":
    main()
