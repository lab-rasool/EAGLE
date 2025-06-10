"""
Visualization utilities for EAGLE with guaranteed color support
"""

import pandas as pd
import numpy as np
from typing import Dict
from lifelines import KaplanMeierFitter


def plot_km_curves(
    risk_df: pd.DataFrame, title: str = None, save_path: str = "km_curves.pdf"
) -> Dict:
    """Plot high-quality Kaplan-Meier curves by risk groups with enhanced information"""
    if title is None:
        title = "Risk-Stratified Survival Curves"

    # Import matplotlib with explicit backend
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    # Reset matplotlib to defaults
    matplotlib.rcdefaults()

    # Create figure with explicit color mode
    fig = plt.figure(figsize=(16, 10), facecolor="white", edgecolor="none")
    gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1], hspace=0.4, figure=fig)
    ax = fig.add_subplot(gs[0], facecolor="white")

    # Define VIBRANT colors using RGB tuples - NO GRAY OR BLACK AND WHITE
    risk_colors = {
        "Low": (0, 200, 83),  # Bright green
        "Low Risk": (0, 200, 83),  # Bright green
        "Medium-Low": (124, 252, 0),  # Lawn green
        "Medium-Low Risk": (124, 252, 0),  # Lawn green
        "Medium": (255, 165, 0),  # Orange
        "Medium Risk": (255, 165, 0),  # Orange
        "Medium-High": (255, 69, 0),  # Orange red
        "Medium-High Risk": (255, 69, 0),  # Orange red
        "High": (220, 20, 60),  # Crimson
        "High Risk": (220, 20, 60),  # Crimson
        "Very High": (139, 0, 0),  # Dark red
        "Very High Risk": (139, 0, 0),  # Dark red
    }

    # Convert to 0-1 range for matplotlib
    risk_colors = {
        k: (v[0] / 255, v[1] / 255, v[2] / 255) for k, v in risk_colors.items()
    }

    kmf_dict = {}
    risk_groups = sorted(risk_df["risk_group"].unique())
    print(f"Risk groups found: {risk_groups}")
    survival_stats = []
    used_colors = {}

    # Plot each risk group using lifelines built-in plotting with explicit colors
    for i, group in enumerate(risk_groups):
        group_data = risk_df[risk_df["risk_group"] == group]

        # Get color for this risk group - NO FALLBACK TO GRAY
        if group in risk_colors:
            color_rgb = risk_colors[group]
        else:
            # Use BRIGHT colors as fallback - NO GRAY
            bright_fallback = [
                (138 / 255, 43 / 255, 226 / 255),  # Blue violet
                (255 / 255, 20 / 255, 147 / 255),  # Deep pink
                (0 / 255, 191 / 255, 255 / 255),  # Deep sky blue
                (255 / 255, 215 / 255, 0 / 255),  # Gold
                (50 / 255, 205 / 255, 50 / 255),  # Lime green
            ]
            color_rgb = bright_fallback[i % len(bright_fallback)]

        used_colors[group] = color_rgb

        # Fit KM model
        kmf = KaplanMeierFitter()
        kmf.fit(
            durations=group_data["survival_time"],
            event_observed=group_data["event"],
            label=f"{group} (n={len(group_data)})",
        )

        # Plot using lifelines plot method with explicit color
        # Use lifelines' internal plotting but override the label afterwards
        kmf.plot_survival_function(
            ax=ax,
            color=color_rgb,
            linewidth=3,
            show_censors=True,
            censor_styles={"marker": "|", "ms": 12, "mew": 2.5},
            ci_show=True,
            ci_alpha=0.2
        )
        
        # Remove labels from the lines that were just plotted
        for line in ax.get_lines()[-2:]:  # Last 2 lines (survival curve and confidence interval)
            line.set_label('_nolegend_')

        # Statistics
        median_survival = kmf.median_survival_time_
        try:
            one_year_survival = float(kmf.survival_function_at_times(12).iloc[0])
        except Exception:
            one_year_survival = None
        try:
            two_year_survival = float(kmf.survival_function_at_times(24).iloc[0])
        except Exception:
            two_year_survival = None

        survival_stats.append(
            {
                "group": group,
                "n": len(group_data),
                "events": int(group_data["event"].sum()),
                "median": median_survival,
                "1yr": one_year_survival,
                "2yr": two_year_survival,
                "color": color_rgb,
            }
        )

        kmf_dict[group] = kmf

    # Force axis limits
    ax.set_xlim(0, risk_df["survival_time"].max() * 1.05)
    ax.set_ylim(-0.05, 1.05)

    # Style with explicit settings
    ax.set_xlabel("Time (months)", fontsize=16, fontweight="bold", color="black")
    ax.set_ylabel("Survival Probability", fontsize=16, fontweight="bold", color="black")
    ax.set_title(title, fontsize=20, fontweight="bold", pad=20, color="black")

    # Grid with light color
    ax.grid(True, alpha=0.3, linestyle="--", color=(0.9, 0.9, 0.9))

    # Spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)

    # Remove legend since the table shows the risk groups with colors and counts
    # Clear any existing legend
    legend = ax.get_legend()
    if legend:
        legend.remove()

    # Add p-value with explicit colors
    if len(risk_groups) > 1:
        from lifelines.statistics import multivariate_logrank_test

        try:
            results = multivariate_logrank_test(
                risk_df["survival_time"], risk_df["risk_group"], risk_df["event"]
            )
            p_text = (
                f"Log-rank p = {results.p_value:.2e}"
                if results.p_value < 0.01
                else f"Log-rank p = {results.p_value:.3f}"
            )
            ax.text(
                0.02,
                0.02,
                p_text,
                transform=ax.transAxes,
                fontsize=12,
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor=(1.0, 0.92, 0.23),  # Yellow
                    edgecolor="black",
                    alpha=0.8,
                ),
            )
        except Exception:
            pass

    # Table subplot
    ax_table = fig.add_subplot(gs[1], facecolor="white")
    ax_table.axis("off")

    # Create table data
    headers = ["Risk Group", "n", "Events", "Median (mo)", "1-yr (%)", "2-yr (%)"]
    table_data = []

    for stat in survival_stats:
        median_str = f"{stat['median']:.1f}" if pd.notna(stat["median"]) else "NR"
        yr1_str = f"{stat['1yr'] * 100:.0f}" if stat["1yr"] is not None else "NA"
        yr2_str = f"{stat['2yr'] * 100:.0f}" if stat["2yr"] is not None else "NA"

        table_data.append(
            [
                stat["group"],
                str(stat["n"]),
                str(stat["events"]),
                median_str,
                yr1_str,
                yr2_str,
            ]
        )

    # Create table using matplotlib table function with explicit colors
    table = ax_table.table(
        cellText=table_data,
        colLabels=headers,
        cellLoc="center",
        loc="center",
        colWidths=[0.2, 0.1, 0.1, 0.15, 0.1, 0.1],
    )

    # Style the table with colors
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)

    # Color the header with light blue instead of gray
    for j in range(len(headers)):
        table[(0, j)].set_facecolor((0.8, 0.9, 1.0))  # Light blue
        table[(0, j)].set_text_props(weight="bold", color="black")
        table[(0, j)].set_edgecolor("black")
        table[(0, j)].set_linewidth(1)

    # Color the risk group cells
    for i in range(len(table_data)):
        for j in range(len(headers)):
            cell = table[(i + 1, j)]
            if j == 0:  # Risk group column
                group_name = table_data[i][0]
                if group_name in used_colors:
                    cell.set_facecolor(used_colors[group_name])
                    cell.set_text_props(color="white", weight="bold")
            else:
                cell.set_facecolor("white")
                cell.set_text_props(color="black")
            cell.set_edgecolor("black")
            cell.set_linewidth(1)

    # Title for table - moved up to avoid overlap
    ax_table.text(
        0.5,
        1.15,  # Moved from 0.95 to 1.15 to be well above the table
        "Risk Group Statistics",
        transform=ax_table.transAxes,
        ha="center",
        va="top",
        fontsize=14,
        fontweight="bold",
        color="black",
    )

    # Force figure to render with colors
    fig.canvas.draw()

    # Save with explicit settings to preserve colors
    plt.savefig(
        save_path,
        format="png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
        pad_inches=0.1,
    )

    # Close figure
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
