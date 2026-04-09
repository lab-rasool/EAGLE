"""
Visualization utilities for EAGLE with guaranteed color support
"""

import pandas as pd
import numpy as np
from typing import Dict
from lifelines import KaplanMeierFitter


def plot_km_curves(
    risk_df: pd.DataFrame, title: str = None, save_path: str = "km_curves.pdf",
    legend_loc: str = "upper right",
) -> Dict:
    """Plot publication-quality Kaplan-Meier curves styled for Nature journals."""
    if title is None:
        title = "Risk-Stratified Survival Curves"

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    matplotlib.rcdefaults()

    # -- Nature journal typography --
    FONT_FAMILY = "Liberation Sans"
    DARK = "#333333"
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [FONT_FAMILY, "DejaVu Sans", "Arial", "Helvetica"],
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "text.color": DARK,
        "axes.labelcolor": DARK,
        "xtick.color": DARK,
        "ytick.color": DARK,
    })

    # -- Lancet color palette (colorblind-safe) --
    risk_colors = {
        "Low": "#0072B2",
        "Low Risk": "#0072B2",
        "Medium-Low": "#56B4E9",
        "Medium-Low Risk": "#56B4E9",
        "Medium": "#E69F00",
        "Medium Risk": "#E69F00",
        "Medium-High": "#D55E00",
        "Medium-High Risk": "#D55E00",
        "High": "#D55E00",
        "High Risk": "#D55E00",
        "Very High": "#CC79A7",
        "Very High Risk": "#CC79A7",
    }
    FALLBACK_COLORS = ["#0072B2", "#D55E00", "#E69F00", "#009E73", "#CC79A7"]

    # -- Layout: curves + at-risk table (no statistics table) --
    n_groups = risk_df["risk_group"].nunique()
    at_risk_height = 0.6 + 0.18 * n_groups
    fig = plt.figure(figsize=(7.5, 5.5 + at_risk_height), facecolor="white")
    gs = gridspec.GridSpec(
        2, 1,
        height_ratios=[5, at_risk_height],
        hspace=0.28,
        figure=fig,
    )
    ax = fig.add_subplot(gs[0], facecolor="white")

    # -- Fit and plot KM curves --
    kmf_dict = {}
    # Order: High -> Medium -> Low (descending risk)
    _risk_order = {
        "Very High Risk": 0, "Very High": 0,
        "High Risk": 1, "High": 1,
        "Medium-High Risk": 2, "Medium-High": 2,
        "Medium Risk": 3, "Medium": 3,
        "Medium-Low Risk": 4, "Medium-Low": 4,
        "Low Risk": 5, "Low": 5,
    }
    risk_groups = sorted(
        risk_df["risk_group"].unique(),
        key=lambda g: _risk_order.get(g, 99),
    )
    used_colors = {}

    for i, group in enumerate(risk_groups):
        group_data = risk_df[risk_df["risk_group"] == group]
        color = risk_colors.get(group, FALLBACK_COLORS[i % len(FALLBACK_COLORS)])
        used_colors[group] = color

        kmf = KaplanMeierFitter()
        kmf.fit(
            durations=group_data["survival_time"],
            event_observed=group_data["event"],
            label=f"{group} (n={len(group_data)})",
        )

        kmf.plot_survival_function(
            ax=ax,
            color=color,
            linewidth=1.5,
            show_censors=True,
            censor_styles={"marker": "+", "ms": 5, "mew": 0.8},
            ci_show=True,
            ci_alpha=0.12,
        )
        kmf_dict[group] = kmf

    # -- Compute numbers at risk --
    max_time = risk_df["survival_time"].max()
    n_timepoints = min(8, max(5, int(max_time / 20) + 1))
    time_points = np.linspace(0, int(max_time), n_timepoints).astype(int)

    at_risk_data = {}
    for group in risk_groups:
        group_data = risk_df[risk_df["risk_group"] == group]
        at_risk_data[group] = [
            int((group_data["survival_time"] >= t).sum()) for t in time_points
        ]

    # -- Axis styling --
    x_max = max_time * 1.05
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Time (months)")
    ax.set_ylabel("Survival probability")
    ax.set_title(title, fontweight="medium", pad=10)

    # Spines: left + bottom only, thin
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(DARK)
        ax.spines[spine].set_linewidth(0.6)

    # Ticks: inward, thin
    ax.tick_params(axis="both", direction="in", length=4, width=0.6, colors=DARK)
    ax.tick_params(axis="x", which="minor", direction="in", length=2, width=0.4)

    # No gridlines
    ax.grid(False)

    # -- Legend: clean, upper right, subtle white backing --
    legend = ax.legend(
        loc=legend_loc,
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.85,
        borderpad=0.4,
        handlelength=1.5,
        handletextpad=0.5,
        labelspacing=0.4,
    )
    for text in legend.get_texts():
        text.set_color(DARK)

    # -- P-value: computed here, drawn in the at-risk section below --
    p_text = None
    if len(risk_groups) > 1:
        from lifelines.statistics import multivariate_logrank_test
        try:
            results = multivariate_logrank_test(
                risk_df["survival_time"], risk_df["risk_group"], risk_df["event"]
            )
            if results.p_value < 0.001:
                p_text = "Log-rank P < 0.001"
            elif results.p_value < 0.01:
                p_text = f"Log-rank P = {results.p_value:.3f}"
            else:
                p_text = f"Log-rank P = {results.p_value:.2f}"
        except Exception:
            pass

    # -- Numbers at risk (text-based, aligned to x-axis) --
    ax_risk = fig.add_subplot(gs[1], facecolor="white")
    ax_risk.set_xlim(ax.get_xlim())
    ax_risk.set_ylim(-0.5, n_groups + 0.3)
    ax_risk.axis("off")

    from matplotlib.transforms import blended_transform_factory
    label_transform = blended_transform_factory(ax_risk.transAxes, ax_risk.transData)

    # Separator line
    ax_risk.axhline(
        y=n_groups - 0.15, xmin=0, xmax=1,
        color=DARK, linewidth=0.4, clip_on=False,
    )

    # Header label
    ax_risk.text(
        -0.04, n_groups + 0.1,
        "No. at risk",
        transform=label_transform,
        ha="right", va="bottom",
        fontsize=7.5, fontweight="bold", color=DARK,
        clip_on=False,
    )

    # P-value: above the separator line, right-aligned
    if p_text is not None:
        ax_risk.text(
            1.0, n_groups + 0.1,
            p_text,
            transform=label_transform,
            ha="right", va="bottom",
            fontsize=8, fontstyle="italic", color=DARK,
            clip_on=False,
        )

    for i, group in enumerate(risk_groups):
        y = n_groups - 1 - i
        # Group label (color-coded, in axes-x space, well left of data)
        ax_risk.text(
            -0.04, y,
            group,
            transform=label_transform,
            ha="right", va="center",
            fontsize=7.5, fontweight="medium",
            color=used_colors[group],
            clip_on=False,
        )
        # Counts at each time point (data coordinates)
        for j, t in enumerate(time_points):
            ax_risk.text(
                t, y,
                str(at_risk_data[group][j]),
                ha="center", va="center",
                fontsize=7.5, color=DARK,
            )

    fig.canvas.draw()

    plt.savefig(
        save_path,
        format="png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
        pad_inches=0.15,
    )
    plt.close(fig)

    return kmf_dict


def create_comprehensive_plots(risk_df: pd.DataFrame, output_dir: str = "."):
    """Create comprehensive visualization plots"""
    import os
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Reset matplotlib
    matplotlib.rcdefaults()

    # Risk score distribution
    fig, ax = plt.subplots(figsize=(10, 6), facecolor="white")
    ax.hist(
        risk_df["risk_score"],
        bins=30,
        alpha=0.8,
        color=(0.1, 0.5, 0.9),  # Nice blue
        edgecolor="black",
    )
    ax.set_xlabel("Risk Score", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Distribution of Risk Scores", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    save_path = os.path.join(output_dir, "risk_score_distribution.png")
    plt.savefig(
        save_path, format="png", dpi=300, bbox_inches="tight", facecolor="white"
    )
    plt.close()

    # Risk score vs survival time scatter
    fig, ax = plt.subplots(figsize=(10, 8), facecolor="white")
    scatter = ax.scatter(
        risk_df["risk_score"],
        risk_df["survival_time"],
        c=risk_df["event"],
        cmap="RdYlBu",
        alpha=0.6,
        s=50,
    )
    plt.colorbar(scatter, label="Event (0=Censored, 1=Death)")
    ax.set_xlabel("Risk Score", fontsize=12)
    ax.set_ylabel("Survival Time (months)", fontsize=12)
    ax.set_title("Risk Score vs Survival Time", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    save_path = os.path.join(output_dir, "risk_vs_survival_scatter.png")
    plt.savefig(
        save_path, format="png", dpi=300, bbox_inches="tight", facecolor="white"
    )
    plt.close()

    # Risk groups distribution
    if "risk_group" in risk_df.columns:
        fig, ax = plt.subplots(figsize=(10, 6), facecolor="white")
        risk_groups = risk_df["risk_group"].value_counts().sort_index()

        # Use VIBRANT colors - NO GRAY
        risk_colors = {
            "Low": (0 / 255, 200 / 255, 83 / 255),
            "Low Risk": (0 / 255, 200 / 255, 83 / 255),
            "Medium-Low": (124 / 255, 252 / 255, 0 / 255),
            "Medium-Low Risk": (124 / 255, 252 / 255, 0 / 255),
            "Medium": (255 / 255, 165 / 255, 0 / 255),
            "Medium Risk": (255 / 255, 165 / 255, 0 / 255),
            "Medium-High": (255 / 255, 69 / 255, 0 / 255),
            "Medium-High Risk": (255 / 255, 69 / 255, 0 / 255),
            "High": (220 / 255, 20 / 255, 60 / 255),
            "High Risk": (220 / 255, 20 / 255, 60 / 255),
            "Very High": (139 / 255, 0 / 255, 0 / 255),
            "Very High Risk": (139 / 255, 0 / 255, 0 / 255),
        }

        colors = []
        for group_name in risk_groups.index:
            if group_name in risk_colors:
                colors.append(risk_colors[group_name])
            else:
                # BRIGHT fallback colors - NO GRAY
                bright_fallback = [
                    (138 / 255, 43 / 255, 226 / 255),  # Blue violet
                    (255 / 255, 20 / 255, 147 / 255),  # Deep pink
                    (0 / 255, 191 / 255, 255 / 255),  # Deep sky blue
                ]
                colors.append(bright_fallback[len(colors) % len(bright_fallback)])

        bars = ax.bar(range(len(risk_groups)), risk_groups.values, color=colors)
        ax.set_xticks(range(len(risk_groups)))
        ax.set_xticklabels(risk_groups.index, rotation=45)
        ax.set_xlabel("Risk Group", fontsize=12)
        ax.set_ylabel("Number of Patients", fontsize=12)
        ax.set_title(
            "Patient Distribution by Risk Group", fontsize=14, fontweight="bold"
        )

        # Add count labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{int(height)}",
                ha="center",
                va="bottom",
            )

        ax.grid(True, alpha=0.3, axis="y")
        save_path = os.path.join(output_dir, "risk_group_distribution.png")
        plt.savefig(
            save_path, format="png", dpi=300, bbox_inches="tight", facecolor="white"
        )
        plt.close()

    # Box plot of risk scores by event status
    fig, ax = plt.subplots(figsize=(8, 6), facecolor="white")
    risk_df_plot = risk_df.copy()
    risk_df_plot["Event Status"] = risk_df_plot["event"].map(
        {0: "Censored", 1: "Death"}
    )

    censored_scores = risk_df_plot[risk_df_plot["Event Status"] == "Censored"][
        "risk_score"
    ]
    death_scores = risk_df_plot[risk_df_plot["Event Status"] == "Death"]["risk_score"]

    box_data = [censored_scores, death_scores]
    positions = [1, 2]

    bp = ax.boxplot(
        box_data,
        positions=positions,
        widths=0.6,
        patch_artist=True,
        labels=["Censored", "Death"],
    )

    # Color the boxes with bright colors
    colors = [(0.5, 0.8, 1.0), (1.0, 0.4, 0.4)]  # Light blue and light red
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)

    ax.set_ylabel("Risk Score", fontsize=12)
    ax.set_title(
        "Risk Score Distribution by Event Status", fontsize=14, fontweight="bold"
    )
    ax.grid(True, alpha=0.3, axis="y")
    save_path = os.path.join(output_dir, "risk_by_event_status.png")
    plt.savefig(
        save_path, format="png", dpi=300, bbox_inches="tight", facecolor="white"
    )
    plt.close()


def plot_dataset_specific(
    risk_df: pd.DataFrame, dataset_name: str, output_dir: str = "."
):
    """Create dataset-specific plots"""
    import os
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from lifelines import KaplanMeierFitter

    if dataset_name == "NSCLC":
        # Stage-specific survival if available
        stage_cols = [col for col in risk_df.columns if col.startswith("STAGE_")]
        if stage_cols:
            for col in stage_cols[:2]:  # Plot first two stage columns
                fig = plt.figure(figsize=(12, 8), facecolor="white")
                ax = fig.add_subplot(111)

                stage_groups = risk_df.groupby(col)
                colors = plt.cm.Set1(np.linspace(0, 1, len(stage_groups)))

                for i, (stage, group) in enumerate(stage_groups):
                    if len(group) >= 5:
                        kmf = KaplanMeierFitter()
                        kmf.fit(
                            durations=group["survival_time"],
                            event_observed=group["event"],
                            label=f"{stage} (n={len(group)})",
                        )
                        kmf.plot_survival_function(ax=ax, color=colors[i], linewidth=2)

                ax.set_xlabel("Time (months)", fontsize=12)
                ax.set_ylabel("Survival Probability", fontsize=12)
                ax.set_title(f"NSCLC Survival by {col}", fontsize=14, fontweight="bold")
                ax.legend(loc="best")
                ax.grid(True, alpha=0.3)
                plt.savefig(
                    os.path.join(output_dir, f"survival_by_{col.lower()}.png"),
                    format="png",
                    dpi=300,
                    bbox_inches="tight",
                    facecolor="white",
                )
                plt.close()

    elif dataset_name == "GBM":
        # Plot by molecular markers if available
        molecular_cols = ["mgmt_methylated", "idh_mutant", "1p19q_codeleted"]
        available_cols = [col for col in molecular_cols if col in risk_df.columns]

        if available_cols:
            fig, axes = plt.subplots(
                1,
                len(available_cols),
                figsize=(6 * len(available_cols), 5),
                facecolor="white",
            )
            if len(available_cols) == 1:
                axes = [axes]

            for i, col in enumerate(available_cols):
                risk_df_plot = risk_df.copy()
                risk_df_plot[col] = risk_df_plot[col].map(
                    {0: "Negative", 1: "Positive"}
                )

                neg_scores = risk_df_plot[risk_df_plot[col] == "Negative"]["risk_score"]
                pos_scores = risk_df_plot[risk_df_plot[col] == "Positive"]["risk_score"]

                bp = axes[i].boxplot(
                    [neg_scores, pos_scores],
                    labels=["Negative", "Positive"],
                    patch_artist=True,
                )

                # Color the boxes with bright colors
                colors = [(1.0, 0.8, 0.8), (0.8, 1.0, 0.8)]  # Light red and light green
                for patch, color in zip(bp["boxes"], colors):
                    patch.set_facecolor(color)

                axes[i].set_title(f"Risk Score by {col.replace('_', ' ').title()}")
                axes[i].set_ylabel("Risk Score")
                axes[i].grid(True, alpha=0.3, axis="y")

            plt.tight_layout()
            plt.savefig(
                os.path.join(output_dir, "gbm_molecular_markers.png"),
                format="png",
                dpi=300,
                bbox_inches="tight",
                facecolor="white",
            )
            plt.close()
