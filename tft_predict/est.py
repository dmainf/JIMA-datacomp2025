import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import math
import os
from scipy.optimize import minimize
from scipy.stats import norm, gamma, pareto, nbinom
from tweedie import tweedie

SHIFT = math.e
ROOT_N = 1.5

print("=== Loading Data ===")
df = pd.read_parquet('data/df_for.parquet')
df_original = df.copy()

df_shift = df.copy()
df_shift['POS販売冊数'] = df_shift['POS販売冊数'] + SHIFT

df_log = df.copy()
df_log['POS販売冊数'] = np.log1p(df_log['POS販売冊数'])

df_root = df.copy()
df_root['POS販売冊数'] = np.power(df_root['POS販売冊数'], 1/ROOT_N)
print("Complete!")

def estimate_tweedie_p(df, transform_name):
    stats = df.groupby('書名', observed=False)['POS販売冊数'].agg(['mean', 'var', 'count'])
    print(f"\n=== Statistics Summary ({transform_name}) ===")
    print(f"Mean: {stats['mean'].mean():.4f}")
    print(f"Median: {stats['mean'].median():.4f}")
    print(f"Std Dev: {np.sqrt(stats['var'].mean()):.4f}")
    print(f"Number of groups: {len(stats)}")

    # log(Var) = p * log(Mean) + log(phi)
    x = np.log(stats['mean'])
    y = np.log(stats['var'])
    x_with_const = sm.add_constant(x)
    model = sm.OLS(y, x_with_const)
    results = model.fit()
    estimated_p = results.params['mean']
    estimated_phi = np.exp(results.params['const'])
    print(f"estimated p : {estimated_p:.4f}")
    print(f"estimated phi: {estimated_phi:.4f}")
    print(f"R-squared  : {results.rsquared:.4f}")
    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, alpha=0.3, s=10)
    plt.plot(x, results.predict(x_with_const), color='r', label=f'Fit: p={estimated_p:.2f}')
    plt.xlabel('log(Mean)')
    plt.ylabel('log(Variance)')
    plt.title(f'Mean-Variance Relationship - {transform_name} (p={estimated_p:.2f})')
    plt.legend()
    plt.grid(True, alpha=0.3)

    output_dir = f'figure/{transform_name}'
    os.makedirs(output_dir, exist_ok=True)

    output_path = f'{output_dir}/{transform_name}_tweedie_estimation.png'
    plt.savefig(output_path)
    print(f"Saved: {output_path}")
    return estimated_p, estimated_phi


def plot_plot(df, transform_type='original', **params):
    estimated_p, estimated_phi = estimate_tweedie_p(df, transform_type)

    print(f"\n=== Generating Distribution Comparison Plot ({transform_type}) ===")
    data = df['POS販売冊数'].dropna()
    mean_val = data.mean()
    std_val = data.std()
    plt.figure(figsize=(10, 6))

    if transform_type == 'original':
        label = 'Actual Distribution'
    elif transform_type == 'shift':
        label = f'Actual Distribution (shift={params["shift"]})'
    elif transform_type == 'log':
        label = 'Actual Distribution (log1p)'
    elif transform_type == 'root':
        label = f'Actual Distribution (root_n={params["root_n"]})'
    else:
        label = 'Actual Distribution'

    if transform_type in ['original', 'shift']:
        bins = np.arange(data.min(), data.max() + 2, 1)
    else:
        bin_width = 0.05
        bins = np.arange(data.min(), data.max() + bin_width, bin_width)
    plt.hist(data, bins=bins, density=True, alpha=0.6, color='blue', label=label)

    x_range = np.linspace(data.min(), data.max(), 1000)


    # Normal Distribution
    plt.plot(x_range, norm.pdf(x_range, mean_val, std_val), 'r-', linewidth=2, label=f'Normal Distribution\n(μ={mean_val:.3f}, σ={std_val:.3f})')

    # Tweedie Distribution
    if estimated_p < 2.0:
        tweedie_pdf = tweedie(p=estimated_p, mu=mean_val, phi=estimated_phi).pdf(x_range)
        plt.plot(x_range, tweedie_pdf, 'g-', linewidth=2, label=f'Tweedie Distribution\n(p={estimated_p:.3f}, φ={estimated_phi:.3f})')
    else:
        print(f"Tweedie distribution skipped (p={estimated_p:.3f} >= 2.0)")

    # Gamma Distribution
    k_gamma = (mean_val ** 2) / (std_val ** 2)
    theta_gamma = (std_val ** 2) / mean_val
    gamma_pdf = gamma.pdf(x_range, k_gamma, scale=theta_gamma)
    plt.plot(x_range, gamma_pdf, 'b-', linewidth=2, label=f'Gamma Distribution\n(k={k_gamma:.3f}, θ={theta_gamma:.3f})')

    # Pareto Distribution
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            data_min = data.min()
            pareto_params = pareto.fit(data, floc=0, fscale=data_min)
            pareto_b = pareto_params[0]
            pareto_loc = pareto_params[1]
            pareto_scale = pareto_params[2]
            pareto_pdf = pareto.pdf(x_range, pareto_b, loc=pareto_loc, scale=pareto_scale)
            plt.plot(x_range, pareto_pdf, 'm-', linewidth=2, label=f'Pareto Distribution\n(b={pareto_b:.3f}, scale={pareto_scale:.3f})')
    except:
        pass

    # === ZINB (Zero-Inflated Negative Binomial) ===
    # ZINB fitting via Maximum Likelihood Estimation
    def zinb_neg_log_likelihood(params, x):
        pi, n, p = params
        # Constraints check
        if not (0 <= pi <= 1 and n > 0 and 0 <= p <= 1):
            return 1e15 # Return huge error if constraints violated

        # P(Y=0) = pi + (1-pi) * NB(0)
        # P(Y=k) = (1-pi) * NB(k) for k > 0

        nb_pmf = nbinom.pmf(x, n, p)

        # Calculate likelihoods
        # For zero values:
        mask_zero = (x == 0)
        prob_zero = pi + (1 - pi) * nbinom.pmf(0, n, p)

        # For non-zero values:
        mask_nonzero = ~mask_zero
        prob_nonzero = (1 - pi) * nb_pmf[mask_nonzero]

        # Combine and sum log-likelihood
        # Add small epsilon to avoid log(0)
        ll = np.zeros_like(x, dtype=float)
        ll[mask_zero] = np.log(prob_zero + 1e-10)
        ll[mask_nonzero] = np.log(prob_nonzero + 1e-10)

        return -np.sum(ll)

    # Prepare data for fitting (ZINB is for count data, so casting to int or rounding is typically needed)
    # Using astype(int) to be consistent with previous logic, but be aware for transformed continuous data
    data_fit = data.astype(int)

    # Initial Guess
    pi_init = (data_fit == 0).mean()
    if std_val**2 > mean_val:
        p_init = mean_val / std_val**2
        n_init = mean_val**2 / (std_val**2 - mean_val)
    else:
        p_init = 0.5
        n_init = 1.0

    try:
        zinb_res = minimize(zinb_neg_log_likelihood, x0=[pi_init, n_init, p_init], args=(data_fit,),
                            bounds=[(0, 1), (1e-5, None), (1e-5, 1)], method='L-BFGS-B')

        pi_est, n_est, p_est = zinb_res.x
        ll_zinb = -zinb_res.fun # Negative of NLL is Log-Likelihood

        # Generate PMF for plotting
        x_int = np.arange(0, int(x_range.max()) + 1)
        nb_pmf_vals = nbinom.pmf(x_int, n_est, p_est)
        zinb_pmf = (1 - pi_est) * nb_pmf_vals
        zinb_pmf[0] += pi_est # Add inflation to zero

        plt.plot(x_int, zinb_pmf, 'orange', linestyle='--', marker='o', markersize=3, alpha=0.7,
                 label=f'ZINB\n($\\pi$={pi_est:.2f}, n={n_est:.2f}, p={p_est:.2f})')
        zinb_success = True
    except Exception as e:
        print(f"ZINB fitting failed: {e}")
        zinb_success = False
        ll_zinb = -np.inf

    print(f"\n=== Model Comparison ({transform_type}) ===")
    n_data = len(data)

    # Normal Distribution
    ll_norm = np.sum(norm.logpdf(data, mean_val, std_val))
    k_norm = 2
    aic_norm = -2 * ll_norm + 2 * k_norm
    bic_norm = -2 * ll_norm + k_norm * np.log(n_data)
    print(f"Normal      - AIC: {aic_norm:.2f}, BIC: {bic_norm:.2f}")

    # Gamma Distribution
    ll_gamma = np.sum(gamma.logpdf(data, k_gamma, scale=theta_gamma))
    k_gamma_params = 2
    aic_gamma = -2 * ll_gamma + 2 * k_gamma_params
    bic_gamma = -2 * ll_gamma + k_gamma_params * np.log(n_data)
    print(f"Gamma       - AIC: {aic_gamma:.2f}, BIC: {bic_gamma:.2f}")

    # Tweedie Distribution
    if estimated_p < 2.0:
        try:
            tweedie_dist = tweedie(p=estimated_p, mu=mean_val, phi=estimated_phi)
            ll_tweedie = np.sum(tweedie_dist.logpdf(data))
            k_tweedie = 2
            aic_tweedie = -2 * ll_tweedie + 2 * k_tweedie
            bic_tweedie = -2 * ll_tweedie + k_tweedie * np.log(n_data)
            print(f"Tweedie     - AIC: {aic_tweedie:.2f}, BIC: {bic_tweedie:.2f}")
        except Exception as e:
            print(f"Tweedie     - Could not calculate: {e}")
    else:
        print(f"Tweedie     - Skipped (p={estimated_p:.3f} >= 2.0)")

    # Pareto Distribution
    try:
        if 'pareto_b' in locals() and not np.isnan(pareto_b):
            ll_pareto = np.sum(pareto.logpdf(data, pareto_b, loc=pareto_loc, scale=pareto_scale))
            k_pareto = 2
            aic_pareto = -2 * ll_pareto + 2 * k_pareto
            bic_pareto = -2 * ll_pareto + k_pareto * np.log(n_data)
            print(f"Pareto      - AIC: {aic_pareto:.2f}, BIC: {bic_pareto:.2f}")
    except:
        pass

    # ZINB Model Stats
    k_zinb = 3 # pi, n, p
    aic_zinb = np.inf
    bic_zinb = np.inf
    if zinb_success:
        aic_zinb = -2 * ll_zinb + 2 * k_zinb
        bic_zinb = -2 * ll_zinb + k_zinb * np.log(n_data)
        print(f"ZINB        - AIC: {aic_zinb:.2f}, BIC: {bic_zinb:.2f}")

    results = {
        'Normal': {'AIC': aic_norm, 'BIC': bic_norm},
        'Gamma': {'AIC': aic_gamma, 'BIC': bic_gamma},
        'ZINB': {'AIC': aic_zinb, 'BIC': bic_zinb}
    }

    if estimated_p < 2.0:
        try:
            results['Tweedie'] = {'AIC': aic_tweedie, 'BIC': bic_tweedie}
        except:
            pass

    try:
        if 'pareto_b' in locals() and not np.isnan(pareto_b):
            results['Pareto'] = {'AIC': aic_pareto, 'BIC': bic_pareto}
    except:
        pass

    if transform_type == 'original':
        xlabel = 'POS Sales'
    elif transform_type == 'shift':
        xlabel = f'POS Sales (shift by {params["shift"]})'
    elif transform_type == 'log':
        xlabel = 'POS Sales (log1p)'
    elif transform_type == 'root':
        xlabel = f'POS Sales ({params["root_n"]}-th root)'
    else:
        xlabel = 'POS Sales'

    plt.xlabel(xlabel)
    plt.ylabel('Density')
    plt.title(f'Distribution Comparison - {transform_type}')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # ========== ZOOM LEFT SIDE (DELETE THIS SECTION TO RESTORE FULL VIEW) ==========
    if transform_type in ['original', 'shift', 'root']:
        max_x = data.max()
        zoom_limit = max_x * 0.01
        plt.xlim(0, zoom_limit)
    # ========== END ZOOM ==========

    output_dir = f'figure/{transform_type}'
    os.makedirs(output_dir, exist_ok=True)

    output_path = f'{output_dir}/{transform_type}_distribution_comparison.png'
    plt.savefig(output_path)
    print(f"Saved: {output_path}")

    plt.figure(figsize=(8, 6))
    plt.boxplot(data.dropna(), vert=True, patch_artist=True,
                whis=[0, 100],
                boxprops=dict(facecolor='lightblue', alpha=0.7),
                medianprops=dict(color='red', linewidth=2),
                whiskerprops=dict(color='blue', linewidth=1.5),
                capprops=dict(color='blue', linewidth=1.5))
    plt.ylabel(xlabel)
    plt.title(f'Boxplot - {transform_type}')
    plt.grid(True, alpha=0.3, axis='y')

    boxplot_path = f'{output_dir}/{transform_type}_boxplot.png'
    plt.savefig(boxplot_path)
    plt.close()
    print(f"Saved: {boxplot_path}")

    return results

all_results = []
all_results.extend([{'Transform': 'original', 'Distribution': dist, **scores}
                    for dist, scores in plot_plot(df_original, transform_type='original').items()])
all_results.extend([{'Transform': 'shift', 'Distribution': dist, **scores}
                    for dist, scores in plot_plot(df_shift, transform_type='shift', shift=SHIFT).items()])
all_results.extend([{'Transform': 'log', 'Distribution': dist, **scores}
                    for dist, scores in plot_plot(df_log, transform_type='log').items()])
all_results.extend([{'Transform': 'root', 'Distribution': dist, **scores}
                    for dist, scores in plot_plot(df_root, transform_type='root', root_n=ROOT_N).items()])

results_df = pd.DataFrame(all_results)

print("\n" + "="*80)
print("=== OVERALL RANKING BY AIC ===")
print("="*80)
results_sorted_aic = results_df.sort_values('AIC')
print(f"\n{'Rank':<6} {'Transform':<15} {'Distribution':<15} {'AIC':<15} {'BIC':<15}")
print("-" * 80)
for rank, (idx, row) in enumerate(results_sorted_aic.iterrows(), 1):
    print(f"{rank:<6} {row['Transform']:<15} {row['Distribution']:<15} {row['AIC']:<15.2f} {row['BIC']:<15.2f}")

print("\n" + "="*80)
print("=== OVERALL RANKING BY BIC ===")
print("="*80)
results_sorted_bic = results_df.sort_values('BIC')
print(f"\n{'Rank':<6} {'Transform':<15} {'Distribution':<15} {'AIC':<15} {'BIC':<15}")
print("-" * 80)
for rank, (idx, row) in enumerate(results_sorted_bic.iterrows(), 1):
    print(f"{rank:<6} {row['Transform']:<15} {row['Distribution']:<15} {row['AIC']:<15.2f} {row['BIC']:<15.2f}")
