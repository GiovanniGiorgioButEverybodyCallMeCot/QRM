import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import cauchy, genextreme, genpareto, laplace, norm, t


# ———————————————————————————————————————————————————————————————————————————————————————————————————————— #
def plot_VaR_estimates(
    dates: pd.DatetimeIndex,
    realized: pd.Series | pd.DataFrame,
    var_norm: pd.DataFrame,  # MultiIndex columns: (alpha, asset)
    var_t: pd.DataFrame,  # MultiIndex columns: (alpha, asset)
    assets: str | list[str] = None,
) -> None:
    """
    Plot realized returns and VaR estimates (Normal and Student-t) over time.
    Shows all alpha levels for each asset in the same subplot.

    Args:
        dates: DatetimeIndex for x-axis.
        realized: Series or DataFrame of realized returns.
        var_norm: DataFrame with MultiIndex columns (alpha, asset) of VaR estimates under Normal.
        var_t: DataFrame with MultiIndex columns (alpha, asset) of VaR estimates under Student-t.
        assets: Single asset name or list of asset names to plot. If None, plots all assets.
    """
    if isinstance(realized, pd.Series):
        realized = realized.to_frame()
    if assets is None:
        assets = realized.columns.tolist()
    elif isinstance(assets, str):
        assets = [assets]

    n = len(assets)
    alphas = var_norm.columns.get_level_values("alpha").unique().tolist()

    ncols = 2 if n > 1 else 1
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
    axes = np.atleast_1d(axes).flatten()

    colormaps = [plt.cm.Blues, plt.cm.Oranges, plt.cm.Greens]

    for i, asset in enumerate(assets):
        ax = axes[i]

        ax.plot(
            dates,
            realized[asset],
            color=".09",
            lw=1.25,
            alpha=0.8,
            label="Realized returns",
            zorder=10,
        )

        for j, alpha in enumerate(alphas):
            cmap = colormaps[j % len(colormaps)]
            color_norm = cmap(0.6)
            color_t = cmap(0.8)

            ax.plot(
                dates,
                -var_norm[alpha][asset],
                color=color_norm,
                lw=1.2,
                linestyle="-",
                label=f"VaR Normal (α={alpha:.2f})",
            )
            ax.plot(
                dates,
                -var_t[alpha][asset],
                color=color_t,
                lw=1.2,
                linestyle="--",
                label=f"VaR t (α={alpha:.2f})",
            )

        ax.axhline(0, color="gray", linestyle=":", lw=0.8)
        ax.set_title(f"{asset}")
        ax.set_ylabel("Return / VaR")
        if i >= (nrows - 1) * ncols:
            ax.set_xlabel("Date")
        ax.legend(fontsize=8, loc="best", ncol=2)
        ax.grid(True, alpha=0.3)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


# ———————————————————————————————————————————————————————————————————————————————————————————————————————— #
def plot_std_resid_diagnostics(Z=None, show: bool = True, save: bool = False) -> None:
    """
    Plot histogram+PDF fits (Normal, Student-t, Laplace, Cauchy)
    and QQ plots for standardized residuals Z (heavy-tail distributions).

    Args:
        Z: Series or array of standardized residuals.
        show: Whether to display the plots.
        save: Whether to save the plots as PNG files.
    """
    mu_hat = Z.mean()
    sig_hat = Z.std(ddof=0)
    df_hat, loc_t, scale_t = t.fit(Z, floc=0)

    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1])
    ax_top = fig.add_subplot(gs[0, :])
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[1, 1])

    xs = np.linspace(Z.min(), Z.max(), 400)
    ax_top.hist(Z, bins=120, density=True, alpha=0.5, label="Z histogram")
    ax_top.plot(xs, norm.pdf(xs, mu_hat, sig_hat), label="Normal")
    ax_top.plot(
        xs, t.pdf(xs, df_hat, loc_t, scale_t), label=f"Student-t (df={df_hat:.1f})"
    )
    ax_top.set_title("Standardized residuals — histogram with fitted distributions")
    ax_top.legend()

    # ———————————————————————————————————————————————————————————————————————————————————————————————————————— #
    def qqplot(ax, sample, q_theor, title):
        s = np.sort(sample)
        q = np.sort(q_theor)
        n = min(len(s), len(q))
        ax.scatter(q[:n], s[:n], s=10, alpha=0.6)
        lo, hi = min(q.min(), s.min()), max(q.max(), s.max())
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
        ax.set_title(title)

    # ———————————————————————————————————————————————————————————————————————————————————————————————————————— #

    n = len(Z)
    p = (np.arange(1, n + 1) - 0.5) / n
    q_norm = norm.ppf(p, mu_hat, sig_hat)
    q_t = t.ppf(p, df_hat, loc_t, scale_t)

    qqplot(ax1, Z, q_norm, "QQ vs Normal")
    qqplot(ax2, Z, q_t, "QQ vs Student-t")

    plt.tight_layout()
    if show:
        plt.show()
    if save:
        plt.savefig("images/std_resid_diagnostics.png", dpi=300)


# ———————————————————————————————————————————————————————————————————————————————————————————————————————— #
def plot_extreme_var(
    dates: pd.DatetimeIndex,
    realized: pd.DataFrame,
    extreme_fits: dict,
    var_extreme: pd.DataFrame,  # MultiIndex: (alpha, asset, theta_type)
    assets: str | list[str] = None,
) -> None:
    """
    Plot realized returns vs extreme value VaR (with/without extremal index).
    Shows all alpha levels for each asset in subplots.
    Works for both GEV and GPD methods.
    Args:
        dates: DatetimeIndex for x-axis.
        realized: DataFrame of realized returns.
        extreme_fits: Dict mapping asset names to fit results from fit_extreme_tail()
        var_extreme: DataFrame with MultiIndex columns (alpha, asset, theta_type) of VaR estimates.
        assets: Single asset name or list of asset names to plot. If None, plots all assets.
    """
    if isinstance(realized, pd.Series):
        realized = realized.to_frame()
    if assets is None:
        assets = realized.columns.tolist()
    elif isinstance(assets, str):
        assets = [assets]

    method = extreme_fits[assets[0]]["method"]

    n = len(assets)
    alphas = var_extreme.columns.get_level_values("alpha").unique().tolist()

    ncols = 2 if n > 1 else 1
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
    axes = np.atleast_1d(axes).flatten()

    if method.upper() == "GEV":
        colormaps = [plt.cm.Purples, plt.cm.Oranges, plt.cm.Greens]
    else:
        colormaps = [plt.cm.Reds, plt.cm.Blues, plt.cm.Greens]

    for i, asset in enumerate(assets):
        ax = axes[i]
        ax.plot(
            dates,
            realized[asset],
            color=".09",
            lw=1.25,
            alpha=0.8,
            label="Realized returns",
            zorder=10,
        )
        for j, alpha in enumerate(alphas):
            cmap = colormaps[j % len(colormaps)]
            color_ind = cmap(0.6)
            color_theta = cmap(0.8)

            ax.plot(
                dates,
                -var_extreme.loc[:, (alpha, asset, "independent")],
                color=color_ind,
                lw=1.2,
                linestyle="-",
                label=f"VaR {method} (θ=1, α={alpha:.2f})",
            )
            ax.plot(
                dates,
                -var_extreme.loc[:, (alpha, asset, "with_theta")],
                color=color_theta,
                lw=1.2,
                linestyle="--",
                label=f"VaR {method} (θ={extreme_fits[asset]['theta']:.2f}, α={alpha:.2f})",
            )

        ax.axhline(0, color="gray", linestyle=":", lw=0.8)
        ax.set_title(f"{asset}")
        ax.set_ylabel("Return / VaR")
        if i >= (nrows - 1) * ncols:
            ax.set_xlabel("Date")
        ax.legend(fontsize=8, loc="best", ncol=2)
        ax.grid(True, alpha=0.3)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


# ———————————————————————————————————————————————————————————————————————————————————————————————————————— #
def plot_extreme_diagnostics(extreme_fits: dict, assets: list[str] = None) -> None:
    """
    QQ-plots for extreme value fit diagnostics.
    Works for both GEV (block maxima) and GPD (excesses).
    Args:
        extreme_fits: Dict mapping asset names to fit results from fit_extreme_tail()
        assets: List of asset names to plot. If None, plots all assets.
    """
    if assets is None:
        assets = list(extreme_fits.keys())
    elif isinstance(assets, str):
        assets = [assets]

    method = extreme_fits[assets[0]]["method"]

    n = len(assets)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 4 * nrows))
    axes = np.atleast_1d(axes).flatten()

    for i, asset in enumerate(assets):
        ax = axes[i]
        fit = extreme_fits[asset]

        if method.upper() == "GEV":
            data_sorted = np.sort(fit["diagnostic_data"].values)
            n_obs = len(data_sorted)
            u = (np.arange(1, n_obs + 1) - 0.5) / n_obs
            theo_q = genextreme.ppf(
                u, fit["gev_shape"], fit["gev_loc"], fit["gev_scale"]
            )
            color = "steelblue"
            ylabel = "Empirical block maxima"
        else:
            data_sorted = np.sort(fit["diagnostic_data"].values)
            n_obs = len(data_sorted)
            u = (np.arange(1, n_obs + 1) - 0.5) / n_obs
            theo_q = genpareto.ppf(u, fit["gpd_xi"], loc=0, scale=fit["gpd_beta"])
            color = "coral"
            ylabel = "Empirical excesses"

        ax.scatter(theo_q, data_sorted, s=18, alpha=0.7, color=color)
        lo = min(theo_q.min(), data_sorted.min())
        hi = max(theo_q.max(), data_sorted.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_title(f"{asset} — {method} QQ-plot")
        ax.set_xlabel(f"{method} quantiles")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


# ———————————————————————————————————————————————————————————————————————————————————————————————————————— #
def plot_evt_garch_var(
    dates: pd.DatetimeIndex,
    realized: pd.DataFrame,
    evt_fits: dict,
    var_evt: pd.DataFrame,  # MultiIndex: (alpha, asset, case)
    assets: str | list[str] = None,
) -> None:
    """
    Plot realized returns vs EVT-GARCH VaR (Cases 3-5).
    Works for both GEV and GPD methods.
    Args:
        dates: DatetimeIndex for x-axis.
        realized: Series or DataFrame of realized returns.
        evt_fits: Dict of in-sample EVT fit parameters per asset.
        var_evt: DataFrame with MultiIndex columns (alpha, asset, case) of VaR estimates.
        assets: Single asset name or list of asset names to plot. If None, plots all assets.
    """
    if isinstance(realized, pd.Series):
        realized = realized.to_frame()
    if assets is None:
        assets = realized.columns.tolist()
    elif isinstance(assets, str):
        assets = [assets]

    n = len(assets)
    alphas = var_evt.columns.get_level_values("alpha").unique().tolist()

    ncols = 2 if n > 1 else 1
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
    axes = np.atleast_1d(axes).flatten()

    colormaps = [plt.cm.Blues, plt.cm.Oranges, plt.cm.Greens]

    for i, asset in enumerate(assets):
        ax = axes[i]
        ax.plot(
            dates,
            realized[asset],
            color=".09",
            lw=1.25,
            alpha=0.8,
            label="Realized returns",
            zorder=10,
        )
        for j, alpha in enumerate(alphas):
            cmap = colormaps[j % len(colormaps)]
            theta_val = evt_fits[asset]["GEV"]["theta"]

            ax.plot(
                dates,
                -var_evt.loc[:, (alpha, asset, "case3")],
                color=cmap(0.5),
                lw=1.2,
                linestyle="-",
                label=f"GEV θ=1 (α={alpha:.2f})",
            )
            ax.plot(
                dates,
                -var_evt.loc[:, (alpha, asset, "case4")],
                color=cmap(0.7),
                lw=1.2,
                linestyle="--",
                label=f"GEV θ={theta_val:.2f} (α={alpha:.2f})",
            )
            ax.plot(
                dates,
                -var_evt.loc[:, (alpha, asset, "case5")],
                color=cmap(0.9),
                lw=1.2,
                linestyle=":",
                label=f"GPD θ=1 (α={alpha:.2f})",
            )

        ax.axhline(0, color="gray", linestyle=":", lw=0.8)
        ax.set_title(f"{asset}")
        ax.set_ylabel("Return / VaR")
        if i >= (nrows - 1) * ncols:
            ax.set_xlabel("Date")
        ax.legend(fontsize=7, loc="best", ncol=1)
        ax.grid(True, alpha=0.3)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


# ———————————————————————————————————————————————————————————————————————————————————————————————————————— #
def plot_var_breach_histogram(
    breach_rates: pd.DataFrame,  # MultiIndex (method, alpha), columns = assets
    alpha_list: list[float],
    assets: str | list[str] = None,
) -> None:
    """
    Create grouped bar chart comparing VaR breach rates across methods.

    Args:
        breach_rates: DataFrame from compute_var_breaches()
        alpha_list: List of confidence levels (for reference lines)
        assets: Single asset or list. If None, plots all assets.
    """
    if assets is None:
        assets = breach_rates.columns.tolist()
    elif isinstance(assets, str):
        assets = [assets]

    methods = breach_rates.index.get_level_values("method").unique().tolist()
    alphas = breach_rates.index.get_level_values("alpha").unique().tolist()

    n = len(assets)
    n_methods = len(methods)
    n_alphas = len(alphas)

    ncols = 2 if n > 1 else 1
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
    axes = np.atleast_1d(axes).flatten()

    x = np.arange(n_methods)
    width = 0.6 / n_alphas

    colours = ["#4899F0", "#F08C48", "#48F0A2", "#F048A2", "#A248F0"]
    hatch_patterns = ["", "///", "\\\\\\", "|||", "---"]

    for i, asset in enumerate(assets):
        ax = axes[i]

        for j, alpha in enumerate(alphas):
            breach_values = []
            for method in methods:
                value = breach_rates.loc[(method, alpha), asset]
                breach_values.append(value)

            offset = (j - n_alphas / 2 + 0.5) * width
            bars = ax.bar(
                x + offset,
                breach_values,
                width,
                label=f"α = {alpha:.2f}" if i == 0 else "",
                color=colours[j],
                edgecolor="black",
                linewidth=0.5,
            )

            for bar in bars:
                bar.set_hatch(hatch_patterns[j % len(hatch_patterns)])

        for j, alpha in enumerate(alphas):
            expected_rate = alpha * 100
            ax.axhline(
                expected_rate,
                color=colours[j],
                linestyle="--",
                linewidth=1,
                alpha=0.8,
                zorder=1,
            )

        ax.set_title(f"{asset}", fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(methods, fontsize=9, rotation=15, ha="right")
        ax.set_ylabel("Breaches (%)", fontsize=10)
        ax.set_xlabel("Model" if i >= (nrows - 1) * ncols else "", fontsize=10)
        ax.grid(True, axis="y", alpha=0.3, linestyle=":")
        ax.set_axisbelow(True)
        if i == 0:
            ax.legend(fontsize=8, loc="upper right")

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


# ———————————————————————————————————————————————————————————————————————————————————————————————————————— #
def plot_breach_magnitude_candles(
    breach_magnitudes: pd.DataFrame,
    alpha_list: list[float],
    assets: str | list[str] = None,
) -> None:
    """
    Create candlestick plot showing breach magnitude distribution.
    """
    if assets is None:
        assets = breach_magnitudes.columns.tolist()
    elif isinstance(assets, str):
        assets = [assets]

    methods = breach_magnitudes.index.get_level_values("method").unique().tolist()
    alphas = breach_magnitudes.index.get_level_values("alpha").unique().tolist()

    n = len(assets)
    n_methods = len(methods)
    n_alphas = len(alphas)

    ncols = 2 if n > 1 else 1
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
    axes = np.atleast_1d(axes).flatten()

    x = np.arange(n_methods)
    width = 0.7 / n_alphas

    colors_box = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    colors_line = ["#185c8d", "#b85c0b", "#195c19", "#651212", "#46315b"]

    for i, asset in enumerate(assets):
        ax = axes[i]
        for j, alpha in enumerate(alphas):
            means, stds, maxs = [], [], []
            for method in methods:
                mean_val = breach_magnitudes.loc[(method, alpha, "mean"), asset]
                std_val = breach_magnitudes.loc[(method, alpha, "std"), asset]
                max_val = breach_magnitudes.loc[(method, alpha, "max"), asset]
                means.append(mean_val)
                stds.append(std_val)
                maxs.append(max_val)

            means = np.array(means)
            stds = np.array(stds)
            maxs = np.array(maxs)

            offset = (j - n_alphas / 2 + 0.5) * width
            positions = x + offset

            for k, pos in enumerate(positions):
                mean = means[k]
                std = stds[k]
                max_val = maxs[k]

                box_bottom = max(0, mean - std)
                box_top = mean + std
                box_height = box_top - box_bottom

                rect = plt.Rectangle(
                    (pos - width / 2, box_bottom),
                    width,
                    box_height,
                    facecolor=colors_box[j],
                    edgecolor="black",
                    linewidth=1.2,
                    alpha=0.7,
                )
                ax.add_patch(rect)

                ax.hlines(
                    mean,
                    pos - width / 2,
                    pos + width / 2,
                    colors="black",
                    linewidth=2,
                    zorder=10,
                )

                ax.vlines(
                    pos,
                    box_top,
                    max_val,
                    colors=colors_line[j],
                    linewidth=1.5,
                    alpha=0.8,
                )
                ax.hlines(
                    max_val,
                    pos - width / 3,
                    pos + width / 3,
                    colors=colors_line[j],
                    linewidth=1.5,
                    alpha=0.8,
                )

            if i == 0:
                ax.plot(
                    [],
                    [],
                    "s",
                    color=colors_box[j],
                    markersize=8,
                    label=f"α = {alpha:.2f}",
                    markeredgecolor="black",
                )

        ax.axhline(0, color="gray", linestyle=":", linewidth=1, alpha=0.5)
        ax.set_title(f"{asset}", fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(methods, fontsize=9, rotation=15, ha="right")
        ax.set_ylabel("Breach Magnitude", fontsize=10)
        ax.grid(True, axis="y", alpha=0.3, linestyle=":")
        ax.set_axisbelow(True)
        if i == 0:
            ax.legend(fontsize=8, loc="upper right")

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


# ———————————————————————————————————————————————————————————————————————————————————————————————————————— #
def plot_mef_and_thresholds(
    returns_dict: dict,
    start_percentile: float = 0.8,
    step: float = 0.01,
    verbose: bool = False,
) -> dict:
    """
    Plot Mean Excess Function (MEF) for each asset and display thresholds for each percentile.
    """
    quantile_grid = np.arange(start_percentile, 1.0, step)
    thresholds_grid = {}

    n = len(returns_dict)
    ncols = 2 if n > 1 else 1
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 4 * nrows))
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, (asset, r) in enumerate(returns_dict.items()):
        r = r.dropna()
        thresholds = r.quantile(quantile_grid)
        thresholds_grid[asset] = thresholds

        excess_means = [
            r[r > u].sub(u).mean() if np.any(r > u) else 0 for u in thresholds
        ]

        ax = axes[idx]
        ax.plot(thresholds, excess_means, marker="o")
        ax.set_xlabel("Threshold u")
        ax.set_ylabel("Mean Excess (E[X-u | X>u])")
        ax.set_title(f"Mean Excess Function — {asset}")
        ax.grid(True)

        if verbose:
            print(f"\nAsset: {asset}")
            for q, val in zip(quantile_grid, thresholds):
                print(f"  Quantile {q:.3f} → Threshold {val:.4f}")

    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.show()

    return thresholds_grid


# ———————————————————————————————————————————————————————————————————————————————————————————————————————— #
def plot_ES_VaR_returns(ES_all, VaR_final, test_all, assets, alphas=[0.01, 0.05]):
    methods = ES_all.columns.get_level_values("method").unique()
    palette = sns.color_palette("tab10", n_colors=len(methods))
    color_dict = dict(zip(methods, palette))

    n_assets = len(assets)
    ncols = 2 if n_assets > 1 else 1
    nrows = math.ceil(n_assets / ncols)

    for alpha in alphas:
        fig, axes = plt.subplots(nrows, ncols, figsize=(16, 5 * nrows))
        if nrows == 1 and ncols == 1:
            axes = [axes]
        elif nrows == 1 or ncols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()

        plot_idx = 0
        for asset in assets:
            ax = axes[plot_idx]
            for method in methods:
                try:
                    es_col = -ES_all.loc[:, (alpha, asset, method)]
                    ax.plot(
                        es_col.index,
                        es_col,
                        linestyle="--",
                        color=color_dict[method],
                        label=f"ES ({method})",
                    )

                    var_col = -VaR_final.loc[:, (alpha, asset, method)]
                    ax.plot(
                        var_col.index,
                        var_col,
                        linestyle="-",
                        color=color_dict[method],
                        label=f"VaR ({method})",
                    )
                except KeyError:
                    print(f"Warning: Missing data for {asset}, {alpha}, {method}")
                    continue

            returns = test_all[asset]
            ax.plot(
                returns.index,
                returns,
                color="0.1",
                alpha=0.7,
                label="Realized returns",
                linewidth=1,
            )

            ax.set_title(f"{asset}")
            ax.set_xlabel("DATE")
            ax.set_ylabel("Value")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right", fontsize="small", frameon=True)

            plot_idx += 1

        for i in range(plot_idx, len(axes)):
            axes[i].axis("off")

        fig.suptitle(
            f"ES, VaR, and Realized Returns - Alpha = {alpha}", fontsize=16, y=1.00
        )
        plt.tight_layout()
        plt.show()
