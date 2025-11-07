import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import seaborn as sns
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from arch import arch_model

UNIVARIATE_GARCH_MODELS = ["GARCH", "GARCH-GJR", "EGARCH", "EGARCH-GJR"]
# ———————————————————————————————————————————————————————————————————————————————————————————————————————————————————————— #
# Descriptive Statistics and Diagnostic Plots


def visual_descriptive_statistics(
    returns_df: pd.DataFrame, plot: bool = True, save: bool = True
) -> pd.DataFrame:
    """
    Computes basic descriptive statistics and generates the following diagnostic plots for each asset's returns in the given DataFrame:
    1. Correlation matrix heatmap between asset returns.
    2. Histograms with normal distribution fit and QQ-plots.
    3. ACF and PACF plots for asset returns.
    4. Rolling moments (mean, variance, skewness, kurtosis) for asset returns.
    5. Highest rolling correlation between any two assets.

    Args:
        returns_df: DataFrame containing asset returns.
        plot: Whether to display the plots.
        save: Whether to save the plots as PNG files.

    Returns:
        DataFrame containing descriptive statistics for each asset's returns.
    """

    rolling_window = 252  # 1 year for daily data
    assets = returns_df.columns
    if save:
        os.makedirs("images/2", exist_ok=True)

    # ————————————————————————————————————————————
    # Correlation matrix heatmap
    # ————————————————————————————————————————————
    corr_matrix = returns_df.corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    plt.figure(figsize=(18, 12))
    sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap="coolwarm",
        annot=True,
        square=True,
        annot_kws={"size": 8, "weight": "bold", "color": "black"},
        cbar_kws={"shrink": 0.75},
    )
    plt.title("Correlation Matrix between Returns", fontsize=18, weight="bold", pad=22)
    plt.xticks(rotation=45, ha="right", fontsize=9, weight="medium")
    plt.yticks(rotation=0, fontsize=9, weight="medium")
    plt.tight_layout()
    if plot:
        plt.show()
    if save:
        plt.savefig("images/correlation_matrix_returns.png", dpi=300)

    # ————————————————————————————————————————————
    # Plot Histograms with Normal Fit and QQ-Plots
    # ————————————————————————————————————————————

    _, axes = plt.subplots(
        nrows=len(assets), ncols=2, figsize=(12, 3 * len(assets))
    )  # height scales with number of assets

    for i, col in enumerate(assets):
        r = returns_df[col]
        # Histogram + Normal fit (left column)
        ax_hist = axes[i, 0]
        ax_hist.hist(r, bins=80, density=True, alpha=0.6)
        x = np.linspace(r.min(), r.max(), 200)
        ax_hist.plot(x, stats.norm.pdf(x, r.mean(), r.std()), lw=2)
        ax_hist.set_title(f"{col} — Histogram & Normal Fit", fontsize=10)
        # QQ-Plot (right column)
        ax_qq = axes[i, 1]
        stats.probplot(r, dist="norm", plot=ax_qq)
        ax_qq.set_title(f"{col} — QQ-Plot", fontsize=10)
    plt.tight_layout()
    if save:
        plt.savefig("images/2/histogram_qqplots.png", dpi=300)
    if plot:
        plt.show()

    # ————————————————————————————————————————————
    # Plot ACF of Returns and Squared Returns
    # ————————————————————————————————————————————
    _, axes_acf = plt.subplots(
        nrows=len(assets), ncols=2, figsize=(12, 3 * len(assets)), squeeze=False
    )
    for i, col in enumerate(assets):
        r = returns_df[col]
        # ACF of returns
        plot_acf(r.dropna(), ax=axes_acf[i, 0], lags=40)
        axes_acf[i, 0].set_title(f"ACF (Returns) — {col}", fontsize=10)
        # ACF of squared returns
        plot_acf((r**2).dropna(), ax=axes_acf[i, 1], lags=40)
        axes_acf[i, 1].set_title(f"ACF (Squared Returns) — {col}", fontsize=10)

    plt.tight_layout()
    if save:
        plt.savefig("images/2/acf_r_r2.png", dpi=300)
    if plot:
        plt.show()

    # ————————————————————————————————————————————
    # Plot PACF of Returns and Squared Returns
    # ————————————————————————————————————————————
    _, axes_pacf = plt.subplots(
        nrows=len(assets), ncols=2, figsize=(12, 3 * len(assets)), squeeze=False
    )
    for i, col in enumerate(assets):
        r = returns_df[col]
        # PACF of returns
        plot_pacf(r.dropna(), ax=axes_pacf[i, 0], lags=40)
        axes_pacf[i, 0].set_title(f"PACF (Returns) — {col}", fontsize=10)
        # PACF of squared returns
        plot_pacf((r**2).dropna(), ax=axes_pacf[i, 1], lags=40)
        axes_pacf[i, 1].set_title(f"PACF (Squared Returns) — {col}", fontsize=10)

    plt.tight_layout()
    if save:
        plt.savefig("images/2/pacf_r_r2.png", dpi=300)
    if plot:
        plt.show()

    # ————————————————————————————————————————————
    # Compute and Plot Rolling Moments
    # ————————————————————————————————————————————

    # Calculate rolling moments
    rolling = {
        "mean": returns_df.rolling(rolling_window).mean(),
        "var": returns_df.rolling(rolling_window).var(),
        "skew": returns_df.rolling(rolling_window).skew(),
        "kurtosis": returns_df.rolling(rolling_window).kurt(),
    }

    # Plot rolling moments
    _, axes = plt.subplots(4, 1, figsize=(14, 14))
    for col in returns_df.columns:
        axes[0].plot(rolling["mean"].index, rolling["mean"][col], lw=1, alpha=0.6)
        axes[1].plot(rolling["var"].index, rolling["var"][col], lw=1.5, alpha=0.65)
        axes[2].plot(rolling["skew"].index, rolling["skew"][col], lw=1.5, alpha=0.65)
        axes[3].plot(
            rolling["kurtosis"].index, rolling["kurtosis"][col], lw=1.5, alpha=0.65
        )

    # Set labels and titles
    axes[0].set_ylabel("Mean")
    axes[1].set_ylabel("Variance")
    axes[2].set_ylabel("Skewness")
    axes[3].set_ylabel("Kurtosis")
    # Format x-axis
    for ax in axes:
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    if save:
        plt.savefig("images/2/rolling_moments.png", dpi=300)
    if plot:
        plt.show()

    # ————————————————————————————————————————————
    # Plot Highest Rolling Correlation Between Any Two Assets
    # ————————————————————————————————————————————

    # Identify the most correlated pair (absolute, excluding diagonal)
    corr = returns_df.corr().abs()
    np.fill_diagonal(corr.values, np.nan)
    asset_1, asset_2 = corr.stack().idxmax()  # pair with highest |corr|

    # Plot rolling correlation
    roll_corr = returns_df[asset_1].rolling(rolling_window).corr(returns_df[asset_2])
    plt.figure(figsize=(10, 4))
    plt.plot(roll_corr)

    # Format x-axis
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # Set labels and title
    plt.title(f"Rolling Correlation (window={rolling_window}) — {asset_1} vs {asset_2}")
    plt.ylabel("Correlation")
    plt.xlabel("Date")
    plt.tick_params(axis="x", rotation=45)
    if save:
        plt.savefig("images/2/highest_rolling_correlation.png", dpi=300)
    if plot:
        plt.show()

    # ————————————————————————————————————————————
    # Descriptive Statistics DataFrame
    # ————————————————————————————————————————————

    descriptive_stat_df = pd.DataFrame(
        {
            "mean": returns_df.mean(),
            "std": returns_df.std(),
            "skew": returns_df.skew(),
            "kurtosis_excess": returns_df.kurtosis(),
            "jarque_bera_p": returns_df.apply(lambda x: stats.jarque_bera(x)[1]),
            "ljungbox_p(10)": returns_df.apply(
                lambda x: acorr_ljungbox(x, lags=[10], return_df=True)[
                    "lb_pvalue"
                ].values[0]
            ),
            "n_extreme_3std": returns_df.apply(
                lambda x: (np.abs(x) > 3 * x.std()).sum()
            ),
        }
    ).round(
        4
    )  # more readable

    return descriptive_stat_df


# ———————————————————————————————————————————————————————————————————————————————————————————————————————————————————————— #
# GARCH-Type Model Fitting and Comparison


def univariate_garch_diagnostics(
    training_set: pd.DataFrame, plot: bool = True, save: bool = True
) -> dict:
    """
    Fits and compares four GARCH-type models: GARCH(1,1), GARCH(1,1)-GJR(1,1), EGARCH(1,1), EGARCH(1,1)-GJR(1,1) for each asset's returns in the given DataFrame.
    Plots the conditional variances and computes average AIC and BIC.
    Args:
        training_set: DataFrame containing asset returns for model fitting.
        plot: Whether to display the plots.
        save: Whether to save the plots as PNG files.
    Returns:
        Dictionary containing average AIC and BIC for each model across all assets.
    """
    # ————————————————————————————————————————————
    # Fit GARCH-type models for each asset
    # ————————————————————————————————————————————
    results = {}
    for asset in training_set.columns:
        # Fit GARCH(1,1)
        garch_results = arch_model(
            training_set[asset].dropna(),
            vol="GARCH",
            p=1,
            q=1,
            dist="Normal",
            mean="Constant",
        ).fit(disp="off")
        # Fit GARCH(1,1)-GJR(1,1)
        garch_gjr_results = arch_model(
            training_set[asset].dropna(),
            vol="GARCH",
            p=1,
            o=1,
            q=1,
            dist="Normal",
            mean="Constant",
        ).fit(disp="off")
        # Fit EGARCH(1,1)
        egarch_results = arch_model(
            training_set[asset].dropna(),
            vol="EGARCH",
            p=1,
            q=1,
            dist="Normal",
            mean="Constant",
        ).fit(disp="off")
        # Fit EGARCH(1,1)-GJR(1,1)
        egarch_gjr_results = arch_model(
            training_set[asset].dropna(),
            vol="EGARCH",
            p=1,
            o=1,
            q=1,
            dist="Normal",
            mean="Constant",
        ).fit(disp="off")
        # Store results
        results[asset] = {
            "GARCH": garch_results,
            "GARCH-GJR": garch_gjr_results,
            "EGARCH": egarch_results,
            "EGARCH-GJR": egarch_gjr_results,
        }

    # ————————————————————————————————————————————
    # Plot conditional variances
    # ————————————————————————————————————————————

    n_rows = (len(training_set.columns) + 1) // 2
    _, axes = plt.subplots(n_rows, 2, figsize=(14, 3 * n_rows))
    axes = axes.flatten()  # Flatten to index easily on 1D

    for i, asset in enumerate(training_set.columns):
        axes[i].plot(
            results[asset]["GARCH-GJR"].conditional_volatility ** 2,
            alpha=0.6,
            label=f"GARCH-GJR",
            color="red",
        )
        axes[i].plot(
            results[asset]["GARCH"].conditional_volatility ** 2,
            alpha=0.6,
            label=f"GARCH",
            color="green",
        )
        axes[i].plot(
            results[asset]["EGARCH-GJR"].conditional_volatility ** 2,
            alpha=0.6,
            label=f"EGARCH-GJR",
            color="yellow",
        )
        axes[i].plot(
            results[asset]["EGARCH"].conditional_volatility ** 2,
            alpha=0.6,
            label=f"EGARCH",
            color="blue",
        )
        axes[i].legend()
        axes[i].set_title(asset)

    # Hide any unused subplots if you have an odd number of assets (we shoulnt't, but just in case)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    if save:
        plt.savefig("images/2/garch_model_comparison.png", dpi=300)
    if plot:
        plt.show()

    # ————————————————————————————————————————————————————————
    # Compute average AIC and BIC across assets for each model
    # ————————————————————————————————————————————————————————

    avg_metrics = {}
    for model_name in UNIVARIATE_GARCH_MODELS:
        aic_values = [results[asset][model_name].aic for asset in training_set.columns]
        bic_values = [results[asset][model_name].bic for asset in training_set.columns]
        avg_metrics[model_name] = [np.mean(aic_values), np.mean(bic_values)]

    return avg_metrics


# ———————————————————————————————————————————————————————————————————————————————————————————————————————————————————————— #
# GARCH-Type Model Fitting and Comparison on Equally Weighted Portfolio


def ewp_garch_diagnostics(
    ewp: pd.Series, plot: bool = True, save: bool = True, print_summaries: bool = False
) -> dict:
    """
    Fits and compares four GARCH-type models: GARCH(1,1), GARCH(1,1)-GJR(1,1), EGARCH(1,1), EGARCH(1,1)-GJR(1,1) for the EWP returns.
    Plots the conditional variances of the models.
    Args:
        ewp: Series containing EWP returns.
        plot: Whether to display the plots.
        save: Whether to save the plots as PNG files.
        print_summaries: Whether to print model summaries.
    Returns:
        Dictionary containing AIC and BIC for each model.
    """

    results = {
        "GARCH": arch_model(
            ewp, vol="Garch", p=1, q=1, dist="Normal", mean="Constant"
        ).fit(disp="off"),
        "GARCH-GJR": arch_model(
            ewp, vol="GARCH", p=1, o=1, q=1, dist="Normal", mean="Constant"
        ).fit(disp="off"),
        "EGARCH": arch_model(
            ewp, vol="EGARCH", p=1, q=1, dist="Normal", mean="Constant"
        ).fit(disp="off"),
        "EGARCH-GJR": arch_model(
            ewp, vol="EGARCH", p=1, o=1, q=1, dist="Normal", mean="Constant"
        ).fit(disp="off"),
    }

    # Portfolio -> plotted singularly because this order is better for visualization
    plt.figure(figsize=(8, 5))
    plt.plot(
        results["GARCH-GJR"].conditional_volatility ** 2,
        alpha=0.7,
        label="GARCH-GJR",
        color="red",
    )
    plt.plot(
        results["GARCH"].conditional_volatility ** 2,
        alpha=0.7,
        label="GARCH",
        color="green",
    )
    plt.plot(
        results["EGARCH-GJR"].conditional_volatility ** 2,
        alpha=0.7,
        label="EGARCH-GJR",
        color="yellow",
    )
    plt.plot(
        results["EGARCH"].conditional_volatility ** 2,
        alpha=0.7,
        label="EGARCH",
        color="blue",
    )
    plt.title("Conditional Variance of Normal Garch (1,1) models - EWP")
    plt.xlabel("Date")
    plt.ylabel("Variance")
    plt.legend()
    plt.tight_layout()
    if save:
        plt.savefig("images/2/ewp_garch_model_comparison.png", dpi=300)
    if plot:
        plt.show()

    if print_summaries:
        for garch_model in UNIVARIATE_GARCH_MODELS:
            print(results[garch_model].summary)
    return {
        garch_model: [results[garch_model].aic, results[garch_model].bic]
        for garch_model in UNIVARIATE_GARCH_MODELS
    }
