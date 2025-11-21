from arch import arch_model
from scipy import stats
from scipy.stats import cauchy, genextreme, genpareto, kstest, laplace, norm, t
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from typing import Dict, Tuple
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


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
    if save:
        plt.savefig("images/correlation_matrix_returns.png", dpi=300)
    if plot:
        plt.show()
    plt.close()
    # ————————————————————————————————————————————
    # Plot Histograms with Normal Fit and QQ-Plots
    # ————————————————————————————————————————————

    fig, axes = plt.subplots(
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
        plt.savefig("images/histogram_qqplots.png", dpi=300)
    if plot:
        plt.show()
    plt.close(fig)

    # ————————————————————————————————————————————
    # Plot ACF of Returns and Squared Returns
    # ————————————————————————————————————————————
    fig, axes_acf = plt.subplots(
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
        plt.savefig("images/acf_r_r2.png", dpi=300)
    if plot:
        plt.show()
    plt.close(fig)

    # ————————————————————————————————————————————
    # Plot PACF of Returns and Squared Returns
    # ————————————————————————————————————————————
    fig, axes_pacf = plt.subplots(
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
        plt.savefig("images/pacf_r_r2.png", dpi=300)
    if plot:
        plt.show()
    plt.close(fig)

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
    fig, axes = plt.subplots(4, 1, figsize=(14, 14))
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
        plt.savefig("images/rolling_moments.png", dpi=300)
    if plot:
        plt.show()
    plt.close(fig)

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
        plt.savefig("images/highest_rolling_correlation.png", dpi=300)
    if plot:
        plt.show()
    plt.close()

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
            "ttest_mean_p": returns_df.apply(lambda x: stats.ttest_1samp(x, 0).pvalue),
        }
    ).round(
        4
    )  # più leggibile

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
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 3 * n_rows))
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
        plt.savefig("images/garch_model_comparison.png", dpi=300)
    if plot:
        plt.show()
    plt.close(fig)

    # ————————————————————————————————————————————————————————
    # Compute average AIC and BIC across assets for each model
    # ————————————————————————————————————————————————————————

    avg_metrics = {}
    for model_name in UNIVARIATE_GARCH_MODELS:
        aic_values = [results[asset][model_name].aic for asset in training_set.columns]
        bic_values = [results[asset][model_name].bic for asset in training_set.columns]
        avg_metrics[model_name] = [np.mean(aic_values), np.mean(bic_values)]

    return avg_metrics, results


def garch_parameter_significance(
    results: dict, training_set: pd.DataFrame, alpha_signif: float = 0.05
) -> pd.DataFrame:
    """
    Calcola la proporzione di parametri significativi per ciascun modello GARCH
    sui dati forniti.

    Args:
        results: Dizionario con risultati dei modelli GARCH per ciascun asset.
                 Deve avere la struttura results[asset][model].
        training_set: DataFrame con i dati dei ritorni degli asset.
        alpha_signif: Soglia di significatività per i p-value (default 0.05).

    Returns:
        DataFrame con proporzione di parametri significativi per modello e parametro.
    """

    # Raccogli tutti i parametri possibili
    all_params = set()
    for asset in training_set.columns:
        for model in UNIVARIATE_GARCH_MODELS:
            all_params.update(results[asset][model].params.index)
    all_params = sorted(list(all_params))

    # Creazione tabella vuota
    significance_table = pd.DataFrame(
        index=UNIVARIATE_GARCH_MODELS, columns=all_params, dtype=float
    )

    # Calcolo proporzione di parametri significativi
    for model_name in UNIVARIATE_GARCH_MODELS:
        model_param_counts = {p: [] for p in all_params}
        for asset in training_set.columns:
            res = results[asset][model_name]
            pvalues = res.pvalues
            for param in all_params:
                if param in pvalues:
                    model_param_counts[param].append(pvalues[param] < alpha_signif)
                else:
                    model_param_counts[param].append(np.nan)
        # Proporzione di parametri significativi (ignorando NaN)
        for param in all_params:
            vals = pd.Series(model_param_counts[param]).dropna()
            significance_table.loc[model_name, param] = (
                vals.mean() if len(vals) > 0 else np.nan
            )

    return significance_table


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
        plt.savefig("images/ewp_garch_model_comparison.png", dpi=300)
    if plot:
        plt.show()
    plt.close()

    if print_summaries:
        for garch_model in UNIVARIATE_GARCH_MODELS:
            print(results[garch_model].summary())

    # AIC / BIC
    aic_bic = {
        model: [results[model].aic, results[model].bic]
        for model in UNIVARIATE_GARCH_MODELS
    }

    # RETURN BOTH
    return aic_bic, results


# ———————————————————————————————————————————————————————————————————————————————————————————————————————————————————————— #
# Rolling Forecasts and VaR Computation Helpers


def _make_model(y: pd.Series, model_name: str = "EGARCH", dist: str = "normal"):
    """
    Helper function to create an arch_model instance based on model_name and dist.
    Args:
        y: Series of returns to model.
        model_name: GARCH-type model name.
        dist: Distribution for innovations.
    Returns:
        arch_model instance.
    """
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
    Args:
        train_series: Series of training returns.
        test_series: Series of test returns.
        model_name: GARCH-type model name.
        dist: Distribution for innovations.
        window: Rolling window size (None for expanding window).
    Returns:
        Tuple of two Series: mu_hat and var_hat.
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


# Plotting functions moved to plotter.py


# ———————————————————————————————————————————————————————————————————————————————————————————————————————————————————————— #


# Plotting functions moved to plotter.py


# ———————————————————————————————————————————————————————————————————————————————————————————————————————————————————————— #


def fit_evt_params_insample(
    returns: pd.Series,
    model_name: str,
    dist: str,
    threshold_quantile: float = 0.95,
    block_freq: str = "M",
) -> dict:
    """
    Fit both GEV and GPD parameters in-sample for EVT-GARCH VaR.
    Args:
        returns: Time series of returns
        model_name: GARCH model name (e.g., 'GARCH')
        dist: Distribution for GARCH (e.g., 'normal')
        threshold_quantile: Quantile for POT threshold (GPD)
        block_freq: Frequency for block maxima (GEV), e.g., 'M' for monthly
    Returns:
        Dict with 'GEV' and 'GPD' parameters including extremal index θ
    """
    # Get standardized residuals
    mdl = _make_model(returns, model_name=model_name, dist=dist)
    res = mdl.fit(disp="off", show_warning=False, tol=1e-6, options={"maxiter": 500})
    Z = pd.Series(res.std_resid, index=returns.index).dropna()
    Y = -Z  # Left tail of Z → right tail of Y

    # Extremal index
    theta_hat = ferro_segers_theta(Y, upper_quantile=0.95)

    # GEV: Block maxima
    Y_mmax = Y.resample(block_freq).max().dropna()
    if len(Y_mmax) < 6:
        Y_mmax = Y.rolling(22, min_periods=1).max().dropna()

    block_sizes = Y.resample(block_freq).size()
    n_days_block = int(max(1, block_sizes.median()))
    c_hat, loc_hat, scale_hat = genextreme.fit(Y_mmax.values)

    # GPD: Peaks over threshold
    u = Y.quantile(threshold_quantile)
    exceed = Y[Y > u]
    excess = (exceed - u).values
    p_u = len(exceed) / max(1, len(Y))

    if len(exceed) >= 20:
        xi_hat, _, beta_hat = genpareto.fit(excess, floc=0.0)
    else:
        xi_hat, beta_hat = 0.0, (excess.mean() if len(exceed) > 0 else 1.0)

    return {
        "GEV": {
            "c": c_hat,
            "loc": loc_hat,
            "scale": scale_hat,
            "n_block": n_days_block,
            "theta": theta_hat,
        },
        "GPD": {
            "u": float(u),
            "p_u": float(p_u),
            "xi": float(xi_hat),
            "beta": float(beta_hat),
            "theta": theta_hat,
        },
    }


# ———————————————————————————————————————————————————————————————————————————————————————————————————————————————————————— #


def compute_evt_garch_var(
    mu: pd.Series,
    sigma2: pd.Series,
    alpha: float,
    evt_params: dict,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Compute VaR combining in-sample EVT quantiles with GARCH forecasts.
    Args:
        mu: Series of conditional mean forecasts.
        sigma2: Series of conditional variance forecasts.
        alpha: Significance level for VaR.
        evt_params: Dict with in-sample EVT fit parameters ('GEV' and 'G
    Returns:
        Tuple of (Case3_GEV_ind, Case4_GEV_theta, Case5_GPD_ind)
    """
    sigma = np.sqrt(sigma2)

    # Case 3: GEV independent (θ=1)
    gev_params = evt_params["GEV"]
    p_ind = (1.0 - alpha) ** gev_params["n_block"]
    y_alpha_ind = genextreme.ppf(
        p_ind, gev_params["c"], gev_params["loc"], gev_params["scale"]
    )
    qZ_gev_ind = -y_alpha_ind
    VaR_case3 = -(mu + sigma * qZ_gev_ind)

    # Case 4: GEV with extremal index
    p_theta = (1.0 - alpha) ** (gev_params["n_block"] * gev_params["theta"])
    y_alpha_theta = genextreme.ppf(
        p_theta, gev_params["c"], gev_params["loc"], gev_params["scale"]
    )
    qZ_gev_theta = -y_alpha_theta
    VaR_case4 = -(mu + sigma * qZ_gev_theta)

    # Case 5: GPD independent (θ=1)
    gpd_params = evt_params["GPD"]
    p_eff = max(gpd_params["p_u"], 1e-12)
    r = max(alpha / p_eff, 1e-12)

    if abs(gpd_params["xi"]) < 1e-8:
        y_alpha_gpd = gpd_params["u"] + gpd_params["beta"] * np.log(1.0 / r)
    else:
        y_alpha_gpd = gpd_params["u"] + (gpd_params["beta"] / gpd_params["xi"]) * (
            r ** (-gpd_params["xi"]) - 1.0
        )

    qZ_gpd_ind = -y_alpha_gpd
    VaR_case5 = -(mu + sigma * qZ_gpd_ind)

    return VaR_case3, VaR_case4, VaR_case5


# ———————————————————————————————————————————————————————————————————————————————————————————————————————————————————————— #


# Plotting functions moved to plotter.py


# ———————————————————————————————————————————————————————————————————————————————————————————————————————————————————————— #


def compute_var_breaches(
    realized: pd.DataFrame,
    var_dict: dict,  # Keys: method names, Values: MultiIndex DataFrames
    alpha_list: list[float],
    assets: str | list[str] = None,
) -> pd.DataFrame:
    """
    Compute VaR breach rates (%) for each method and alpha level.

    Args:
        realized: DataFrame of realized returns
        var_dict: Dict mapping method names to VaR DataFrames
        alpha_list: List of confidence levels
        assets: Assets to analyze. If None, uses all assets.

    Returns:
        DataFrame with MultiIndex (method, alpha) and columns as assets
    """
    if isinstance(realized, pd.Series):
        realized = realized.to_frame()

    if assets is None:
        assets = realized.columns.tolist()
    elif isinstance(assets, str):
        assets = [assets]

    results = []

    for method, var_data in var_dict.items():
        # how many levels in the VaR columns?
        nlevels = getattr(getattr(var_data, "columns", None), "nlevels", 1)

        for alpha in alpha_list:
            breach_rates = {}

            for asset in assets:

                # --- 1. Select the appropriate VaR series for this method/asset/alpha ---
                if nlevels == 3:
                    # MultiIndex: (alpha, asset, something)
                    level2_vals = list(var_data.columns.levels[2])

                    # Choose the "main" label in the 3rd level depending on method
                    if method in ("GEV", "GPD"):
                        # Prefer dependence-corrected version
                        if "with_theta" in level2_vals:
                            chosen = "with_theta"
                        elif "GEV_theta" in level2_vals:
                            chosen = "GEV_theta"
                        else:
                            chosen = level2_vals[-1]  # fallback

                    elif method in ("EVT", "EVT-GARCH"):
                        # EVT-GARCH VaR: prefer the dependence-corrected case
                        if "case4" in level2_vals:
                            chosen = "case4"
                        elif "GEV_theta" in level2_vals:
                            chosen = "GEV_theta"
                        elif "with_theta" in level2_vals:
                            chosen = "with_theta"
                        elif "GEV_ind" in level2_vals and "GEV_theta" in level2_vals:
                            chosen = "GEV_theta"
                        elif "GEV_theta" in level2_vals:
                            chosen = "GEV_theta"
                        else:
                            chosen = level2_vals[-1]  # fallback
                    else:
                        # Any other 3-level structure: just take the last label
                        chosen = level2_vals[-1]

                    var_series = var_data.loc[:, (alpha, asset, chosen)]

                else:
                    # 2-level case: (alpha, asset) → Normal / t etc.
                    var_series = var_data.loc[:, (alpha, asset)]

                # If, for any reason, we still got a DataFrame, take the first column
                if isinstance(var_series, pd.DataFrame):
                    var_series = var_series.iloc[:, 0]

                # --- 2. Align indices before comparison (very important) ---
                r_asset, v_asset = realized[asset].align(var_series, join="inner")

                # --- 3. Compute breach rate: realized return < -VaR (loss exceeds VaR) ---
                breaches = (r_asset < -v_asset).sum()
                breach_rate = (
                    100.0 * breaches / len(r_asset) if len(r_asset) > 0 else np.nan
                )
                breach_rates[asset] = breach_rate

            results.append({"method": method, "alpha": alpha, **breach_rates})

    df = pd.DataFrame(results)
    df = df.set_index(["method", "alpha"])
    return df


# ———————————————————————————————————————————————————————————————————————————————————————————————————————————————————————— #


# Plotting functions moved to plotter.py


# ———————————————————————————————————————————————————————————————————————————————————————————————————————————————————————— #


def compute_breach_magnitudes(
    realized: pd.DataFrame,
    var_dict: dict,
    alpha_list: list[float],
    assets: str | list[str] = None,
) -> pd.DataFrame:
    """
    Compute breach magnitude statistics: mean, std, and max of (VaR - realized)
    when realized < -VaR (i.e., losses exceed VaR).
    Args:
        realized: DataFrame of realized returns
        var_dict: Dict mapping method names to VaR DataFrames
        alpha_list: List of confidence levels
        assets: Assets to analyze. If None, uses all assets.
    Returns:
        DataFrame with MultiIndex (method, alpha, stat) and columns as assets
    """
    if isinstance(realized, pd.Series):
        realized = realized.to_frame()

    if assets is None:
        assets = realized.columns.tolist()
    elif isinstance(assets, str):
        assets = [assets]

    results = []

    for method, var_data in var_dict.items():
        for alpha in alpha_list:
            stats = {}

            for asset in assets:
                # Extract VaR series
                if "theta_type" in var_data.columns.names:
                    var_series = var_data.loc[:, (alpha, asset, "with_theta")]
                elif "case" in var_data.columns.names:
                    var_series = var_data.loc[:, (alpha, asset, "case4")]
                else:
                    var_series = var_data.loc[:, (alpha, asset)]

                # Breach magnitude: |realized - (-VaR)| when realized < -VaR
                breach_mask = realized[asset] < -var_series

                if breach_mask.sum() > 0:
                    # Difference: how much the loss exceeded VaR
                    breach_diff = (
                        -var_series[breach_mask] - realized[asset][breach_mask]
                    ).abs()

                    mean_breach = breach_diff.mean()
                    std_breach = breach_diff.std()
                    max_breach = breach_diff.max()
                else:
                    mean_breach = 0.0
                    std_breach = 0.0
                    max_breach = 0.0

                if asset not in stats:
                    stats[asset] = {}

                stats[asset]["mean"] = mean_breach
                stats[asset]["std"] = std_breach
                stats[asset]["max"] = max_breach

            # Add rows for mean, std, max
            for stat in ["mean", "std", "max"]:
                row = {"method": method, "alpha": alpha, "stat": stat}
                for asset in assets:
                    row[asset] = stats[asset][stat]
                results.append(row)

    df = pd.DataFrame(results)
    df = df.set_index(["method", "alpha", "stat"])
    return df


# ———————————————————————————————————————————————————————————————————————————————————————————————————————————————————————— #


# Plotting functions moved to plotter.py


# ———————————————————————————————————————————————————————————————————————————————————————————————————————————————————————— #
def compute_all_ES(
    mu_all,
    sigma2_all,
    df_params,
    gev_fits,
    gpd_fits,
    VaR_gev,
    VaR_gpd,
    alpha_list,
    assets_list,
    test_dates,
):
    """
    Compute Expected Shortfall (ES) for all assets and methods.
    Args:
        mu_all: DataFrame of conditional mean forecasts for all assets.
        sigma2_all: DataFrame of conditional variance forecasts for all assets.
        df_params: Dict mapping asset names to degrees of freedom for Student-t.
        gev_fits: Dict mapping asset names to GEV fit parameters.
        gpd_fits: Dict mapping asset names to GPD fit parameters.
        VaR_gev: DataFrame with MultiIndex columns (alpha, asset, theta_type) of GEV VaR estimates.
        VaR_gpd: DataFrame with MultiIndex columns (alpha, asset, method) of GPD VaR estimates.
        alpha_list: List of confidence levels.
        assets_list: List of asset names.
        test_dates: DatetimeIndex for the test period.
    Returns:
        DataFrame with MultiIndex columns (alpha, asset, method) of ES estimates."""

    all_assets = list(assets_list) + ["EWP"]
    sigma_all = np.sqrt(sigma2_all)

    ES = pd.DataFrame(
        index=test_dates,
        columns=pd.MultiIndex.from_product(
            [alpha_list, all_assets, ["Normal", "Student-t", "GEV_theta", "GPD_ind"]],
            names=["alpha", "asset", "method"],
        ),
        dtype=float,
    )

    # =============================
    # NORMAL ES
    # =============================
    for alpha in alpha_list:
        z = -norm.pdf(norm.ppf(alpha)) / alpha
        for asset in all_assets:
            ES.loc[:, (alpha, asset, "Normal")] = (
                mu_all[asset] - sigma_all[asset] * z  # ES is negative
            )

    # =============================
    # STUDENT-T ES
    # =============================
    for alpha in alpha_list:
        for asset in all_assets:
            nu = df_params[asset]
            t_alpha = t.ppf(alpha, nu)
            scale_term = (nu + t_alpha**2) / (nu - 1)
            t_pdf = t.pdf(t_alpha, nu)
            ES_t_std = -t_pdf * scale_term / alpha
            ES.loc[:, (alpha, asset, "Student-t")] = (
                mu_all[asset] - sigma_all[asset] * ES_t_std
            )

    # =============================
    # GEV–THETA ES
    # =============================
    for alpha in alpha_list:
        for asset in all_assets:
            try:
                VaR_theta = VaR_gev.loc[:, (alpha, asset, "with_theta")]
                xi = gev_fits[asset]["gev_shape"]
                beta = gev_fits[asset]["gev_scale"]

                if abs(xi) > 1e-6:
                    ES_gev = VaR_theta + (
                        beta - xi * (VaR_theta - gev_fits[asset]["gev_loc"])
                    ) / (1 - xi)
                else:
                    ES_gev = VaR_theta - beta * np.log(alpha)

                ES.loc[:, (alpha, asset, "GEV_theta")] = ES_gev
            except:
                ES.loc[:, (alpha, asset, "GEV_theta")] = np.nan

    # =============================
    # GPD–IND ES
    # =============================
    for alpha in alpha_list:
        for asset in all_assets:

            # Seleziona VaR GPD
            VaR_g = VaR_gpd.loc[:, (alpha, asset, "independent")]

            # Parametri GPD
            xi = gpd_fits[asset]["gpd_xi"]
            beta = gpd_fits[asset]["gpd_beta"]
            threshold = gpd_fits[asset]["threshold"]

            # Se xi non è definito, metti NaN
            if np.isnan(xi):
                ES.loc[:, (alpha, asset, "GPD_ind")] = np.nan
                continue

            # Calcolo ES (solo se xi < 1)
            if xi < 1:
                ES_gpd = (VaR_g + beta - xi * (VaR_g - threshold)) / (1 - xi)
            else:
                ES_gpd = np.nan

            # Salva nell'ES DataFrame
            ES.loc[:, (alpha, asset, "GPD_ind")] = ES_gpd

    return ES


# ———————————————————————————————————————————————————————————————————————————————————————————————————————————————————————— #


# Plotting functions moved to plotter.py


# ———————————————————————————————————————————————————————————————————————————————————————————————————————————————————————— #
def explore_gpd_thresholds(
    all_assets,
    all_returns,
    start_percentile=0.8,
    step=0.005,
    max_threshold=0.99,
    verbose: bool = False,
) -> dict:
    """
    Explore different thresholds for GPD fit and print xi for each asset and threshold.

    Args:
        all_assets: list of asset names
        all_returns: dict of asset_name -> returns series
        start_percentile: starting quantile for threshold (default 0.8)
        step: increment in quantile for threshold
        max_threshold: maximum quantile to try (default 0.99)
    """
    import numpy as np
    from scipy.stats import genpareto

    thresholds = np.arange(start_percentile, max_threshold + step, step)

    results = {asset: {} for asset in all_assets}

    for asset in all_assets:
        r = all_returns[asset].dropna()
        print(f"\nAsset: {asset}")
        for q in thresholds:
            u = np.quantile(r, q)
            exceed = r[r > u]
            excess = exceed - u

            if len(exceed) < 1:
                xi_hat = np.nan
            else:
                xi_hat, loc_hat, beta_hat = genpareto.fit(excess.values, floc=0.0)
            results[asset][q] = xi_hat
            print(f"  Threshold quantile {q:.4f} (u={u:.4f}): xi = {xi_hat:.4f}")

    return results


# ———————————————————————————————————————————————————————————————————————————————————————————————————————————————————————— #


# Duplicate of compute_all_ES removed; single definition kept above


# ———————————————————————————————————————————————————————————————————————————————————————————————————————————————————————— #


# Plotting functions moved to plotter.py


# ———————————————————————————————————————————————————————————————————————————————————————————————————————————————————————— #


def gpd_var_manual_thresholds(
    thresholds_manual: dict,
    all_returns: dict,
    test_dates: pd.DatetimeIndex,
    alpha_list: list,
    mu_all: pd.DataFrame,
    sigma_all: pd.DataFrame,
    test_all: pd.DataFrame,
) -> tuple[dict, pd.DataFrame]:
    """
    Run GPD-based extreme value VaR analysis using manually specified thresholds
    with dynamic VaR based on GARCH mu/sigma (ignores extremal index).

    Args:
        thresholds_manual: dict of asset_name -> threshold value
        all_returns: dict of asset -> standardized residuals (for fitting)
        test_dates: DatetimeIndex for test period
        alpha_list: list of confidence levels
        mu_all: GARCH mean forecasts (all assets including EWP)
        sigma_all: GARCH sigma forecasts (all assets including EWP)
        test_all: test returns DataFrame (all assets including EWP)

    Returns:
        Tuple of (gpd_fit_results_dict, VaR_DataFrame)
    """
    all_assets = list(all_returns.keys())

    # 1) Fit GPD models
    print("Fitting GPD models with manual thresholds...")
    gpd_fits = {}

    for asset in all_assets:
        r = all_returns[asset]
        u = thresholds_manual.get(asset, 0.0)
        exceed = r[r > u]
        excess = exceed - u

        if len(exceed) < 1:
            print(f"Warning: {asset} has no exceedances above threshold {u:.4f}")
            xi_hat, beta_hat, p_u, n_exc = np.nan, np.nan, 0, 0
        else:
            xi_hat, _, beta_hat = genpareto.fit(excess.values, floc=0.0)
            p_u = len(exceed) / len(r)
            n_exc = len(exceed)

        gpd_fits[asset] = {
            "gpd_xi": xi_hat,
            "gpd_beta": beta_hat,
            "threshold": u,
            "p_u": p_u,
            "n_exceedances": n_exc,
            "method": "GPD",
            "diagnostic_data": excess,
        }

        print(
            f"{asset}: u={u:.4f}, ξ={xi_hat:.4f}, β={beta_hat:.4f}, n_exc={n_exc} (p_u={p_u:.4f})"
        )

    # 2) Compute dynamic VaR forecasts
    VaR_extreme = pd.DataFrame(
        index=test_dates,
        columns=pd.MultiIndex.from_product(
            [alpha_list, all_assets, ["independent"]],
            names=["alpha", "asset", "theta_type"],
        ),
        dtype=float,
    )

    print("\nComputing GPD-based dynamic VaR forecasts...")
    for alpha in alpha_list:
        for asset in all_assets:
            gpd = gpd_fits[asset]
            if np.isnan(gpd["gpd_xi"]):
                VaR_extreme.loc[:, (alpha, asset, "independent")] = np.nan
            else:
                xi, beta, u_val, p_u_val = (
                    gpd["gpd_xi"],
                    gpd["gpd_beta"],
                    gpd["threshold"],
                    gpd["p_u"],
                )
                factor = ((1 / alpha * p_u_val) ** xi - 1) / xi
                VaR_extreme.loc[:, (alpha, asset, "independent")] = mu_all[
                    asset
                ] + sigma_all[asset] * (u_val + beta * factor)

    # 3) Summary statistics
    summary_base = pd.DataFrame(
        {
            "xi": [gpd_fits[a]["gpd_xi"] for a in all_assets],
            "n_exceedances": [gpd_fits[a]["n_exceedances"] for a in all_assets],
            "threshold": [gpd_fits[a]["threshold"] for a in all_assets],
        },
        index=all_assets,
    )

    for alpha in alpha_list:
        var_ind_means = [
            VaR_extreme.loc[:, (alpha, a, "independent")].mean() for a in all_assets
        ]
        alpha_summary = pd.DataFrame(
            {"VaR_ind_mean": var_ind_means}, index=all_assets
        ).join(summary_base)
        print(f"\nα = {alpha}\n" + "—" * 80)
        print(alpha_summary)
    # 4) Visualization
    n = len(all_assets)
    ncols = 2 if n > 1 else 1
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
    axes = np.atleast_1d(axes).flatten()

    colormaps = [plt.cm.Reds, plt.cm.Blues, plt.cm.Greens]

    for i, asset in enumerate(all_assets):
        ax = axes[i]
        ax.plot(
            test_dates,
            test_all[asset],
            color="#1a1a1a",
            lw=1.25,
            alpha=0.8,
            label="Realized returns",
            zorder=10,
        )

        for j, alpha in enumerate(alpha_list):
            color = colormaps[j % len(colormaps)](0.6)
            ax.plot(
                test_dates,
                -VaR_extreme.loc[:, (alpha, asset, "independent")],
                color=color,
                lw=1.2,
                label=f"VaR (α={alpha:.2f})",
            )

        ax.axhline(0, color="gray", linestyle=":", lw=0.8, alpha=0.6)
        ax.set_title(asset, fontweight="bold")
        ax.set_ylabel("Return / VaR")
        ax.legend(fontsize=8, loc="best", ncol=2, framealpha=0.9)
        ax.grid(True, alpha=0.25, linestyle="--")

        if i >= (nrows - 1) * ncols:
            ax.set_xlabel("Date")

    for j in range(n, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

    return gpd_fits, VaR_extreme


# ———————————————————————————————————————————————————————————————————————————————————————————————————————————————————————— #
