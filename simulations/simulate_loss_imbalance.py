"""Compare lossless, uniformly lossy, and imbalanced 3x3 reachable regions.

The model uses an ideal reciprocal Fourier MMI and reflectors
``r_j = sqrt(R_j) exp(i phi_j)`` with fixed power returns ``R_j``.  Output
powers are shown after normalization by their sum so that geometric changes
can be separated from insertion loss.
"""

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from tfln_paths import OUTPUT_ROOT


import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = OUTPUT_ROOT / "simulations"
OUTPUT_PNG = OUTPUT_DIR / "reachable_region_loss_3x3.png"
OUTPUT_SVG = OUTPUT_DIR / "reachable_region_loss_3x3.svg"
DETAIL_PNG = OUTPUT_DIR / "reachable_region_imbalance_detail_3x3.png"
DETAIL_SVG = OUTPUT_DIR / "reachable_region_imbalance_detail_3x3.svg"

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 8,
        "axes.titlesize": 9,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
)

BLUE = "#3378A8"
ORANGE = "#D97722"
GRAY = "#777777"


def fourier_mmi() -> np.ndarray:
    indices = np.arange(3)
    return np.exp(-2j * np.pi * np.outer(indices, indices) / 3) / np.sqrt(3)


def reachable_powers(
    power_returns: np.ndarray, samples: int = 321
) -> tuple[np.ndarray, np.ndarray]:
    """Return absolute and normalized powers for input port zero."""
    phase = np.linspace(0, 2 * np.pi, samples, endpoint=False)
    phi_1, phi_2 = np.meshgrid(phase, phase, indexing="ij")
    reflectors = np.stack(
        (
            np.full(phi_1.size, np.sqrt(power_returns[0]), dtype=complex),
            np.sqrt(power_returns[1]) * np.exp(1j * phi_1.ravel()),
            np.sqrt(power_returns[2]) * np.exp(1j * phi_2.ravel()),
        ),
        axis=1,
    )

    mmi = fourier_mmi()
    input_field = np.array([1.0, 0.0, 0.0], dtype=complex)
    arm_field = mmi @ input_field
    reflected = reflectors * arm_field[None, :]
    output_field = reflected @ mmi
    powers = np.abs(output_field) ** 2
    normalized = powers / powers.sum(axis=1, keepdims=True)
    return powers, normalized


def barycentric_xy(power: np.ndarray) -> np.ndarray:
    return np.column_stack(
        (power[:, 1] + 0.5 * power[:, 2], np.sqrt(3) * power[:, 2] / 2)
    )


def ideal_boundary(samples: int = 1201) -> np.ndarray:
    phase = np.linspace(0, 2 * np.pi, samples)
    powers = (
        np.column_stack(
            (
                (1 + 2 * np.cos(phase)) ** 2,
                (1 + 2 * np.cos(phase - 2 * np.pi / 3)) ** 2,
                (1 + 2 * np.cos(phase + 2 * np.pi / 3)) ** 2,
            )
        )
        / 9
    )
    return barycentric_xy(powers)


def routed_metrics(power_returns: np.ndarray) -> dict[str, float]:
    """Metrics at phases that align all three arm fields at one output."""
    amplitudes = np.sqrt(power_returns)
    total = power_returns.mean()
    target = amplitudes.sum() ** 2 / 9
    normalized_target = target / total
    strongest_leak = (total - target) / 2
    extinction_db = (
        np.inf if strongest_leak <= 0 else 10 * np.log10(target / strongest_leak)
    )
    return {
        "total_return": total,
        "insertion_loss_db": -10 * np.log10(total),
        "normalized_target": normalized_target,
        "extinction_db": extinction_db,
    }


def make_imbalance_detail(imbalanced: np.ndarray) -> None:
    """Show the deformation, vertex exclusion, and scaling with imbalance."""
    _, normalized = reachable_powers(imbalanced, samples=501)
    xy = barycentric_xy(normalized)
    boundary_xy = ideal_boundary()
    metrics = routed_metrics(imbalanced)
    target = metrics["normalized_target"]
    leak = (1 - target) / 2

    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.75), constrained_layout=False)
    fig.subplots_adjust(left=0.06, right=0.97, bottom=0.19, top=0.78, wspace=0.38)
    fig.suptitle(
        "How fixed arm-loss imbalance changes ideal $3\\times3$ routing",
        fontsize=10,
        y=0.96,
    )

    # (a) Complete normalized region overlaid on the lossless region.
    ax = axes[0]
    ax.fill(
        boundary_xy[:, 0],
        boundary_xy[:, 1],
        color="#D9D9D9",
        alpha=0.85,
        label="lossless region",
        zorder=0,
    )
    ax.hexbin(
        xy[:, 0],
        xy[:, 1],
        gridsize=145,
        extent=(-0.02, 1.02, -0.02, 0.89),
        mincnt=1,
        cmap=mpl.colors.ListedColormap([BLUE]),
        alpha=0.58,
        linewidths=0,
        rasterized=True,
        zorder=1,
    )
    ax.plot(boundary_xy[:, 0], boundary_xy[:, 1], color=GRAY, ls="--", lw=1.0)
    simplex = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2], [0.0, 0.0]])
    ax.plot(simplex[:, 0], simplex[:, 1], color="#555555", lw=1.0)
    ax.text(-0.02, -0.035, r"$P_1$", ha="center", va="top")
    ax.text(1.02, -0.035, r"$P_2$", ha="center", va="top")
    ax.text(0.5, np.sqrt(3) / 2 + 0.03, r"$P_3$", ha="center", va="bottom")
    ax.text(
        0.5,
        0.43,
        "blue: 0/1/2 dB imbalance\ngray: lossless",
        ha="center",
        va="center",
        fontsize=6.7,
    )
    ax.set_title("a) Region deformation", loc="left", pad=3)
    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(-0.09, 0.96)
    ax.set_aspect("equal")
    ax.axis("off")

    # (b) Magnified view of the missing P1 routing vertex.
    ax = axes[1]
    near_vertex = normalized[:, 0] > 0.94
    leakage_xy = 100 * normalized[near_vertex, 1:3]
    ax.hexbin(
        leakage_xy[:, 0],
        leakage_xy[:, 1],
        gridsize=95,
        extent=(0, 6, 0, 6),
        mincnt=1,
        cmap=mpl.colors.ListedColormap([BLUE]),
        alpha=0.68,
        linewidths=0,
        rasterized=True,
    )
    ax.scatter(
        [0], [0], marker="*", s=55, color="black", label="lossless vertex", zorder=5
    )
    ax.scatter(
        [100 * leak],
        [100 * leak],
        marker="o",
        s=34,
        color=ORANGE,
        edgecolor="white",
        linewidth=0.6,
        label="best imbalanced state",
        zorder=6,
    )
    ax.annotate(
        f"({100 * leak:.3f}%, {100 * leak:.3f}%)",
        xy=(100 * leak, 100 * leak),
        xytext=(1.25, 1.45),
        arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.8),
        color=ORANGE,
        fontsize=6.8,
    )
    ax.set_title(r"b) Zoom near $P_1=1$", loc="left", pad=3)
    ax.set_xlabel(r"Leakage to $P_2$ (%)")
    ax.set_ylabel(r"Leakage to $P_3$ (%)")
    ax.set_xlim(-0.12, 5.2)
    ax.set_ylim(-0.12, 5.2)
    ax.set_aspect("equal")
    ax.grid(color="#DDDDDD", lw=0.5)
    ax.legend(loc="upper right", frameon=False, fontsize=6.2, handletextpad=0.3)

    # (c) Routing penalty for a continuous loss-spread sweep [0, Delta/2, Delta].
    ax = axes[2]
    spread_db = np.linspace(0.02, 8.0, 500)
    returns = 10 ** (
        -np.column_stack((np.zeros_like(spread_db), spread_db / 2, spread_db)) / 10
    )
    amplitudes = np.sqrt(returns)
    routed_fraction = amplitudes.sum(axis=1) ** 2 / (3 * returns.sum(axis=1))
    leakage_percent = 100 * (1 - routed_fraction)
    ax.plot(spread_db, leakage_percent, color=BLUE, lw=1.8)
    ax.fill_between(spread_db, leakage_percent, 1e-4, color=BLUE, alpha=0.12)
    example_leakage = 100 * (1 - target)
    ax.scatter(
        [2], [example_leakage], color=ORANGE, s=30, edgecolor="white", lw=0.6, zorder=5
    )
    ax.annotate(
        f"2 dB span\n{example_leakage:.3f}% total leakage\nER = {metrics['extinction_db']:.1f} dB",
        xy=(2, example_leakage),
        xytext=(3.05, 0.24),
        arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.8),
        color=ORANGE,
        fontsize=6.8,
    )
    ax.set_title("c) Routing penalty", loc="left", pad=3)
    ax.set_xlabel("Arm-loss span (dB)")
    ax.set_ylabel("Minimum total leakage (\%)")
    ax.set_yscale("log")
    ax.set_xlim(0, 8)
    ax.set_ylim(1e-4, 20)
    ax.grid(color="#DDDDDD", lw=0.5, which="both")

    fig.savefig(DETAIL_PNG, dpi=300)
    fig.savefig(DETAIL_SVG)
    plt.close(fig)


def main() -> None:
    cases = [
        ("Lossless", np.array([1.0, 1.0, 1.0]), 0.90),
        ("Equal 1 dB loss", np.full(3, 10 ** (-1 / 10)), 0.43),
        ("Imbalanced: 0, 1, 2 dB", 10 ** (-np.array([0.0, 1.0, 2.0]) / 10), 0.58),
    ]

    simplex = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2], [0.0, 0.0]])
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.0), constrained_layout=False)
    boundary_xy = ideal_boundary()
    blue_map = mpl.colors.ListedColormap([BLUE])

    normalized_sets: list[np.ndarray] = []
    for panel, (title, returns, intensity) in enumerate(cases):
        powers, normalized = reachable_powers(returns)
        normalized_sets.append(normalized)
        xy = barycentric_xy(normalized)
        metrics = routed_metrics(returns)

        ax = axes[panel]
        ax.hexbin(
            xy[:, 0],
            xy[:, 1],
            gridsize=125,
            extent=(-0.02, 1.02, -0.02, 0.89),
            mincnt=1,
            cmap=blue_map,
            alpha=intensity,
            linewidths=0,
            rasterized=True,
        )
        ax.plot(simplex[:, 0], simplex[:, 1], color=GRAY, lw=1.0)
        ax.plot(
            boundary_xy[:, 0],
            boundary_xy[:, 1],
            color=GRAY if panel == 2 else BLUE,
            lw=1.0,
            ls="--" if panel == 2 else "-",
            alpha=0.9,
            zorder=4,
        )
        ax.scatter(simplex[:3, 0], simplex[:3, 1], s=9, color="black", zorder=5)
        ax.text(-0.02, -0.035, r"$P_1$", ha="center", va="top")
        ax.text(1.02, -0.035, r"$P_2$", ha="center", va="top")
        ax.text(0.5, np.sqrt(3) / 2 + 0.035, r"$P_3$", ha="center", va="bottom")
        ax.set_title(f"{chr(97 + panel)}) {title}", loc="left", pad=4)

        er_text = (
            "infinite"
            if np.isinf(metrics["extinction_db"])
            else f'{metrics["extinction_db"]:.1f} dB'
        )
        if panel == 2:
            target = metrics["normalized_target"]
            leak = (1 - target) / 2
            routed = np.array(
                [[target, leak, leak], [leak, target, leak], [leak, leak, target]]
            )
            routed_xy = barycentric_xy(routed)
            ax.scatter(
                routed_xy[:, 0],
                routed_xy[:, 1],
                s=24,
                marker="o",
                facecolor=ORANGE,
                edgecolor="white",
                linewidth=0.55,
                zorder=6,
            )
            ax.text(
                0.5,
                0.56,
                "gray dashed: lossless boundary\norange: best routing states",
                ha="center",
                va="center",
                fontsize=6.6,
                color="#555555",
            )
        metric_text = (
            f"total return = {metrics['total_return']:.3f} "
            f"(IL {metrics['insertion_loss_db']:.2f} dB)\n"
            f"max target = {100 * metrics['normalized_target']:.2f}%; "
            f"ER = {er_text}"
        )
        ax.text(
            0.5,
            -0.10,
            metric_text,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=6.8,
        )
        ax.set_xlim(-0.09, 1.09)
        ax.set_ylim(-0.13, 0.96)
        ax.set_aspect("equal")
        ax.axis("off")

    uniform_difference = np.max(np.abs(normalized_sets[0] - normalized_sets[1]))
    fig.suptitle(
        "Ideal $3\\times3$ Fourier MMI: normalized reachable output powers",
        fontsize=10,
        y=0.96,
    )
    fig.subplots_adjust(left=0.035, right=0.985, bottom=0.20, top=0.79, wspace=0.25)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=300)
    fig.savefig(OUTPUT_SVG)
    plt.close(fig)
    make_imbalance_detail(cases[2][1])

    print(f"uniform_normalized_max_difference={uniform_difference:.3e}")
    for title, returns, _ in cases:
        metrics = routed_metrics(returns)
        print(
            f"{title}: returns={returns.tolist()}, "
            f"sumP={metrics['total_return']:.6f}, "
            f"IL={metrics['insertion_loss_db']:.6f} dB, "
            f"target_fraction={metrics['normalized_target']:.6f}, "
            f"ER={metrics['extinction_db']:.6f} dB"
        )


if __name__ == "__main__":
    main()
