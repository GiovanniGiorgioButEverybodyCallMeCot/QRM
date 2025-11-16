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
from scipy.stats import t, kstest, norm, genextreme
from typing import Tuple, Dict
from scipy.stats import genpareto

# ——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————— #

UNIVARIATE_GARCH_MODELS = ["GARCH", "GARCH-GJR", "EGARCH", "APARCH"]

# ——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————— #
# GARCH-Type Model Fitting Helper Function


def fit_garch_models(
    ewp: pd.Series, distribution: str = "Normal", exclude: list = []
) -> dict:
    """
    Fits various GARCH-type models to the given equally weighted portfolio (EWP) returns.
    Args:
        ewp: Series containing EWP returns.
        exclude: List of model names to exclude from fitting.
    Returns:
        Dictionary containing fitted models.
    """
    model_specs = {
        "GARCH": {"vol": "GARCH", "p": 1, "q": 1},
        "GARCH-GJR": {"vol": "GARCH", "p": 1, "o": 1, "q": 1},
        "EGARCH": {"vol": "EGARCH", "p": 1, "o": 1, "q": 1},
        "APARCH": {"vol": "APARCH", "p": 1, "o": 1, "q": 1},
    }

    # I know that fitting models one by one is more readable, but how cool is this dictionary comprehension?
    return {
        name: arch_model(ewp, **specs, dist=distribution, mean="Constant").fit(
            disp="off"
        )
        for name, specs in model_specs.items()
        if name not in exclude
    }


# ———————————————————————————————————————————————————————————————————————————————————————————————————————————————————————— #
# Descriptive Statistics and Diagnostic Plots


def visual_descriptive_statistics(
    returns_df: pd.DataFrame, plot: bool = False, save: bool = False
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
        # Store results
        results[asset] = fit_garch_models(training_set[asset])

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
            results[asset]["APARCH"].conditional_volatility ** 2,
            alpha=0.6,
            label=f"APARCH",
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

    results = fit_garch_models(ewp)

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
        results["APARCH"].conditional_volatility ** 2,
        alpha=0.7,
        label="APARCH",
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
    }  # levereging dictionary comprehension because it looks nice


# ———————————————————————————————————————————————————————————————————————————————————————————————————————————————————————— #
# Rolling Forecasts and VaR Computation Helpers


def _make_model(y: pd.Series, model_name: str = "EGARCH", dist: str = "normal"):
    dist = dist.lower()
    if model_name.upper() == "GARCH":
        return arch_model(y, mean="Constant", vol="GARCH", p=1, q=1, dist=dist)
    elif model_name.upper() == "EGARCH":
        return arch_model(y, mean="Constant", vol="EGARCH", p=1, o=1, q=1, dist=dist)
    elif model_name.upper() == "GARCH-GJR":
        return arch_model(y, mean="Constant", vol="GARCH", p=1, o=1, q=1, dist=dist)
    elif model_name.upper() == "APARCH":
        return arch_model(y, mean="Constant", vol="APARCH", p=1, o=1, q=1, dist=dist)
    else:
        raise ValueError("Unknown model_name")


# ———————————————————————————————————————————————————————————————————————————————————————————————————————————————————————— #
# Rolling Forecasts
def rolling_forecast_sigma(
    train_series: pd.Series,
    test_series: pd.Series,
    model_name: str = "EGARCH",
    dist: str = "normal",
    window: int | None = 1000,
) -> Tuple[pd.Series, pd.Series]:
    """
    For each date in test_series, fit the model on past data up to t-1 and forecast t.
    Returns two Series indexed by test_series.index: mu_hat and var_hat.
    """
    # Build a continuous series to slice by position safely
    full = pd.concat([train_series, test_series]).dropna()
    n_train = len(train_series)
    test_dates = test_series.index
    n_forecast = len(test_dates)

    mu_f = pd.Series(index=test_dates, dtype=float)
    var_f = pd.Series(index=test_dates, dtype=float)

    for i in range(n_forecast):
        end_pos = n_train + i  # position of day BEFORE the forecasted day
        if window is None:
            start_pos = 0
        else:
            start_pos = max(0, end_pos - window)

        y_insample = full.iloc[start_pos:end_pos].dropna()
        if len(y_insample) < 60:
            mu_f.iloc[i] = np.nan
            var_f.iloc[i] = np.nan
            continue

        try:
            mdl = _make_model(y_insample, model_name=model_name, dist=dist)
            res = mdl.fit(
                disp="off", show_warning=False, tol=1e-6, options={"maxiter": 500}
            )
            fc = res.forecast(horizon=1, reindex=False)
            mu_f.iloc[i] = float(fc.mean.values[-1, 0])
            var_f.iloc[i] = float(fc.variance.values[-1, 0])
        except Exception:
            mu_f.iloc[i] = np.nan
            var_f.iloc[i] = np.nan

    return mu_f, var_f


# ———————————————————————————————————————————————————————————————————————————————————————————————————————————————————————— #
# Residual Distribution Fitting
def fit_residual_distribution(
    train_series: pd.Series, model_name: str = "EGARCH", dist_for_fit: str = "normal"
) -> Dict[str, float]:
    """
    Fit the chosen GARCH on the training set with Normal innovations.
    Extract standardized residuals z and test Normal vs. Student-t;
    return estimated df for t and KS p-values.
    Args:
        train_series: Series of training returns.
        model_name: GARCH-type model name.
        dist_for_fit: Distribution for fitting the GARCH model.
    Returns:
        Dictionary with keys 'df_t', 'ks_norm_p', 'ks_t_p'.
    """
    out = {"df_t": np.nan, "ks_norm_p": np.nan, "ks_t_p": np.nan}
    try:
        mdl = _make_model(
            train_series.dropna(), model_name=model_name, dist=dist_for_fit
        )
        res = mdl.fit(
            disp="off", show_warning=False, tol=1e-6, options={"maxiter": 500}
        )
        z = pd.Series(res.std_resid).dropna().values

        # estimate Student-t df on z (loc=0)
        df_t, _, _ = t.fit(z, floc=0)
        ks_norm_p = kstest(z, "norm", args=(np.mean(z), np.std(z, ddof=0))).pvalue
        ks_t_p = kstest(z, "t", args=(df_t, 0, 1)).pvalue

        out.update({"df_t": df_t, "ks_norm_p": ks_norm_p, "ks_t_p": ks_t_p})
    except Exception:
        pass
    return out


# ———————————————————————————————————————————————————————————————————————————————————————————————————————————————————————— #
# VaR Computation


def compute_VaR(
    mu_forecast: pd.Series | pd.DataFrame,
    var_forecast: pd.Series | pd.DataFrame,
    alpha: float,
    distribution: str = "normal",
    df_param: float | dict | None = None,
):
    """
    Returns VaR as a positive number:  VaR = -(mu + sqrt(var) * q_alpha)

    Args:
        mu_forecast: Series or DataFrame of forecasted means.
        var_forecast: Series or DataFrame of forecasted variances.
        alpha: Significance level for VaR.
        distribution: 'normal' or 'student-t'.
        df_param: Degrees of freedom for Student-t (float or dict mapping columns to df values).
    Returns:
        Series or DataFrame of VaR values.
    """
    s = np.sqrt(var_forecast)

    if distribution == "normal":
        q = norm.ppf(alpha)
    elif distribution == "student-t":
        if df_param is None:
            raise ValueError("df_param required for student-t VaR")
        if np.isscalar(df_param):
            q = t.ppf(alpha, df_param)
        else:
            # dict case: create Series with appropriate alignment
            if isinstance(mu_forecast, pd.DataFrame):
                q = pd.Series(
                    {
                        col: t.ppf(alpha, df_param.get(col, np.nan))
                        for col in mu_forecast.columns
                    }
                )
            else:
                q = t.ppf(alpha, df_param)
    else:
        raise ValueError("distribution must be 'normal' or 'student-t'")

    return -(mu_forecast + s * q)


# ———————————————————————————————————————————————————————————————————————————————————————————————————————————————————————— #
# Plotting Helpers


# Might've over-engineered this a bit, but now it accepts both single and multiple assets plotting and more alpha levels
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
    # Normalize realized to DataFrame
    if isinstance(realized, pd.Series):
        realized = realized.to_frame()
    # Determine which assets to plot
    if assets is None:
        assets = realized.columns.tolist()
    elif isinstance(assets, str):
        assets = [assets]

    n = len(assets)
    alphas = var_norm.columns.get_level_values("alpha").unique().tolist()

    # Subplot layout
    ncols = 2 if n > 1 else 1
    nrows = int(np.ceil(n / ncols))
    # Create subplots
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
    axes = np.atleast_1d(axes).flatten()

    # Define 3 colormaps that cycle
    colormaps = [plt.cm.Blues, plt.cm.Oranges, plt.cm.Greens]

    # Plot each asset
    for i, asset in enumerate(assets):
        ax = axes[i]

        # Plot realized returns
        ax.plot(
            dates,
            realized[asset],
            color=".09",
            lw=1.25,
            alpha=0.8,
            label="Realized returns",
            zorder=10,
        )

        # Plot VaR for each alpha level
        for j, alpha in enumerate(alphas):
            # Assign colormap cyclically
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

        # x-labels only at bottom row
        if i >= (nrows - 1) * ncols:
            ax.set_xlabel("Date")

        ax.legend(fontsize=8, loc="best", ncol=2)
        ax.grid(True, alpha=0.3)

    # Remove unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


# ———————————————————————————————————————————————————————————————————————————————————————————————————————————————————————— #
# Standardized Residual Diagnostics Plotting
def plot_std_resid_diagnostics(Z=None, show: bool = True, save: bool = False) -> None:
    """
    Plot histogram+PDF fits (Normal, t, GEV) and QQ plots (3 panels) for standardized residuals Z.
    Args:
        Z: Series of standardized residuals.
        show: Whether to display the plots.
        save: Whether to save the plots as PNG files.
    Returns:
        None
    """

    # --- fit distributions ---
    mu_hat = Z.mean()
    sig_hat = Z.std(ddof=0)
    df_hat, loc_t, scale_t = t.fit(Z, floc=0)
    c_hat, loc_gev, scale_gev = genextreme.fit(Z)

    params = {
        "normal": {"mu": mu_hat, "sigma": sig_hat},
        "t": {"df": df_hat, "loc": loc_t, "scale": scale_t},
        "gev": {"c": c_hat, "loc": loc_gev, "scale": scale_gev},
    }

    # --- create figure layout ---
    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.2, 1])
    ax_top = fig.add_subplot(gs[0, :])
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[1, 1])
    ax3 = fig.add_subplot(gs[1, 2])

    # --- histogram + PDFs ---
    xs = np.linspace(Z.min(), Z.max(), 400)
    ax_top.hist(Z, bins=120, density=True, alpha=0.5, label="Z histogram")
    ax_top.plot(xs, norm.pdf(xs, mu_hat, sig_hat), label="Normal fit")
    ax_top.plot(xs, t.pdf(xs, df_hat, loc_t, scale_t), label=f"t fit (df={df_hat:.1f})")
    ax_top.plot(
        xs,
        genextreme.pdf(xs, c_hat, loc_gev, scale_gev),
        label=f"GEV fit (c={c_hat:.2f})",
    )
    ax_top.set_title("Standardized residuals — histogram with fitted distributions")
    ax_top.legend()

    # --- local QQ helper ---
    def qqplot(ax, sample, q_theor, title):
        s = np.sort(sample)
        q = np.sort(q_theor)
        n = min(len(s), len(q))
        ax.scatter(q[:n], s[:n], s=10, alpha=0.6)
        lo, hi = min(q.min(), s.min()), max(q.max(), s.max())
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
        ax.set_title(title)

    # --- QQ theoretical quantiles ---
    n = len(Z)
    p = (np.arange(1, n + 1) - 0.5) / n
    q_norm = norm.ppf(p, mu_hat, sig_hat)
    q_t = t.ppf(p, df_hat, loc_t, scale_t)
    q_gev = genextreme.ppf(p, c_hat, loc_gev, scale_gev)

    # --- QQ plots ---
    qqplot(ax1, Z, q_norm, "QQ vs Normal")
    qqplot(ax2, Z, q_t, "QQ vs Student-t")
    qqplot(ax3, Z, q_gev, "QQ vs GEV")
    plt.tight_layout()

    if show:
        plt.show()

    if save:
        plt.savefig("image/5/std_resid_diagnostics.png", dpi=300)


# ———————————————————————————————————————————————————————————————————————————————————————————————————————————————————————— #
def ferro_segers_theta(series: pd.Series, upper_quantile: float = 0.95) -> float:
    """
    Estimate extremal index θ using Ferro-Segers-style approach.

    Args:
        series: Time series of data (e.g., standardized residuals)
        upper_quantile: Threshold quantile for exceedances (default: 0.95)

    Returns:
        Estimated extremal index θ ∈ (0, 1]
    """
    u = series.quantile(upper_quantile)
    exc_idx = series[series > u].index

    if len(exc_idx) < 2:
        return 1.0

    # Convert datetime index to day counts
    if isinstance(exc_idx, pd.DatetimeIndex):
        inter_days = exc_idx.to_series().diff().dt.total_seconds().dropna() / 86400.0
    else:
        inter_days = pd.Series(np.diff(np.asarray(exc_idx, dtype=float)))

    if len(inter_days) == 0:
        return 1.0

    m = inter_days.mean()
    v = inter_days.var(ddof=1) if len(inter_days) > 1 else 0.0

    # Ferro-Segers shrinkage to [0,1]
    theta = m / (m + (v / max(m, 1e-12)))
    return float(np.clip(theta, 0.01, 1.0))


# ———————————————————————————————————————————————————————————————————————————————————————————————————————————————————————— #


def fit_extreme_tail(
    returns: pd.Series,
    model_name: str,
    dist: str,
    method: str = "GEV",
    threshold_quantile: float = 0.95,
    upper_quantile: float = 0.95,
) -> dict:
    """
    Fit extreme value distribution to standardized residuals (left tail).

    Args:
        returns: Time series of returns
        model_name: GARCH model name (e.g., 'GARCH')
        dist: Distribution for GARCH (e.g., 'normal')
        method: 'GEV' for Block Maxima or 'GPD' for Peaks-Over-Threshold
        threshold_quantile: Quantile for POT threshold (GPD only)
        upper_quantile: Quantile for extremal index estimation

    Returns:
        Dictionary with distribution parameters, extremal index, and diagnostic data
    """
    # Fit GARCH model to get standardized residuals
    mdl = _make_model(returns, model_name=model_name, dist=dist)
    res = mdl.fit(disp="off", show_warning=False, tol=1e-6, options={"maxiter": 500})
    Z = pd.Series(res.std_resid, index=returns.index).dropna()

    # Left tail analysis: Y = -Z (large Y = large losses)
    Y = -Z

    # Estimate extremal index
    theta_hat = ferro_segers_theta(Y, upper_quantile=upper_quantile)

    result = {
        "method": method,
        "theta": theta_hat,
    }

    if method.upper() == "GEV":
        # Block Maxima approach
        Y_mmax = Y.resample("M").max().dropna()
        block_sizes = Y.resample("M").size()
        n_days_block = int(block_sizes.median())

        # Fit GEV to block maxima
        c_hat, loc_hat, scale_hat = genextreme.fit(Y_mmax.values)

        result.update(
            {
                "gev_shape": c_hat,
                "gev_loc": loc_hat,
                "gev_scale": scale_hat,
                "n_blocks": len(Y_mmax),
                "n_days_block": n_days_block,
                "diagnostic_data": Y_mmax,  # For QQ plots
            }
        )

    elif method.upper() == "GPD":
        # Peaks-Over-Threshold approach
        u = Y.quantile(threshold_quantile)
        exceed = Y[Y > u]
        excess = exceed - u

        n = len(Y)
        n_exc = len(exceed)
        p_u = n_exc / n

        if n_exc < 30:
            print(f"  Warning: Only {n_exc} exceedances — estimates may be unstable")

        # Fit GPD to excesses (loc=0)
        xi_hat, loc_hat, beta_hat = genpareto.fit(excess.values, floc=0.0)

        result.update(
            {
                "gpd_xi": xi_hat,
                "gpd_beta": beta_hat,
                "threshold": u,
                "threshold_quantile": threshold_quantile,
                "p_u": p_u,
                "n_exceedances": n_exc,
                "diagnostic_data": excess,  # For QQ plots
            }
        )
    else:
        raise ValueError(f"Unknown method: {method}. Use 'GEV' or 'GPD'")

    return result


# ———————————————————————————————————————————————————————————————————————————————————————————————————————————————————————— #


def compute_extreme_var(
    mu: pd.Series,
    sigma2: pd.Series,
    alpha: float,
    extreme_params: dict,
) -> tuple[pd.Series, pd.Series]:
    """
    Compute VaR using extreme value theory (GEV or GPD) with and without extremal index.

    Args:
        mu: Conditional mean forecasts
        sigma2: Conditional variance forecasts
        alpha: Confidence level (e.g., 0.05)
        extreme_params: Dictionary with fit results from fit_extreme_tail()

    Returns:
        Tuple of (VaR_independent, VaR_with_theta)
    """
    method = extreme_params["method"]
    theta = extreme_params["theta"]
    sigma = np.sqrt(sigma2)

    if method.upper() == "GEV":
        # GEV-based VaR
        c = extreme_params["gev_shape"]
        loc = extreme_params["gev_loc"]
        scale = extreme_params["gev_scale"]
        n_days = extreme_params["n_days_block"]

        # Map daily α-quantile via block GEV
        p_block_ind = (1.0 - alpha) ** n_days
        p_block_theta = (1.0 - alpha) ** (n_days * theta)

        y_alpha_ind = genextreme.ppf(p_block_ind, c, loc, scale)
        y_alpha_theta = genextreme.ppf(p_block_theta, c, loc, scale)

    elif method.upper() == "GPD":
        # GPD-based VaR
        u = extreme_params["threshold"]
        p_u = extreme_params["p_u"]
        xi = extreme_params["gpd_xi"]
        beta = extreme_params["gpd_beta"]

        def gpd_quantile(alpha_tail, theta_val=1.0):
            p_eff = max(theta_val * p_u, 1e-12)
            r = max(alpha_tail / p_eff, 1e-12)

            if abs(xi) < 1e-8:  # Exponential limit
                return u + beta * np.log(1.0 / r)
            else:
                return u + (beta / xi) * (r ** (-xi) - 1.0)

        y_alpha_ind = gpd_quantile(alpha, theta_val=1.0)
        y_alpha_theta = gpd_quantile(alpha, theta_val=theta)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Convert to residual quantiles: q_Z(α) = -y_alpha
    qZ_ind = -y_alpha_ind
    qZ_theta = -y_alpha_theta

    # VaR = -(μ + σ * q_Z)
    VaR_ind = -(mu + sigma * qZ_ind)
    VaR_theta = -(mu + sigma * qZ_theta)

    return VaR_ind, VaR_theta


# ———————————————————————————————————————————————————————————————————————————————————————————————————————————————————————— #


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
    """
    if isinstance(realized, pd.Series):
        realized = realized.to_frame()

    if assets is None:
        assets = realized.columns.tolist()
    elif isinstance(assets, str):
        assets = [assets]

    # Determine method from first asset
    method = extreme_fits[assets[0]]["method"]

    n = len(assets)
    alphas = var_extreme.columns.get_level_values("alpha").unique().tolist()

    # Subplot layout
    ncols = 2 if n > 1 else 1
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
    axes = np.atleast_1d(axes).flatten()

    # Color schemes by method
    if method.upper() == "GEV":
        colormaps = [plt.cm.Purples, plt.cm.Oranges, plt.cm.Greens]
    else:  # GPD
        colormaps = [plt.cm.Reds, plt.cm.Blues, plt.cm.Greens]

    for i, asset in enumerate(assets):
        ax = axes[i]

        # Plot realized returns
        ax.plot(
            dates,
            realized[asset],
            color=".09",
            lw=1.25,
            alpha=0.8,
            label="Realized returns",
            zorder=10,
        )

        # Plot VaR for each alpha
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

    # Remove unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


# ———————————————————————————————————————————————————————————————————————————————————————————————————————————————————————— #


def plot_extreme_diagnostics(extreme_fits: dict, assets: list[str] = None) -> None:
    """
    QQ-plots for extreme value fit diagnostics.
    Works for both GEV (block maxima) and GPD (excesses).
    """
    if assets is None:
        assets = list(extreme_fits.keys())
    elif isinstance(assets, str):
        assets = [assets]

    # Determine method from first asset
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
            # GEV QQ-plot (block maxima)
            data_sorted = np.sort(fit["diagnostic_data"].values)
            n_obs = len(data_sorted)
            u = (np.arange(1, n_obs + 1) - 0.5) / n_obs
            theo_q = genextreme.ppf(
                u, fit["gev_shape"], fit["gev_loc"], fit["gev_scale"]
            )

            color = "steelblue"
            ylabel = "Empirical block maxima"

        else:  # GPD
            # GPD QQ-plot (excesses)
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
