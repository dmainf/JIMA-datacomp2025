import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import math
import os

SHIFT = 1
ROOT_N = 1.5

print("=== Loading Data ===")
df = pd.read_parquet('data/df_for.parquet')
df_original = df.copy()

df_shift = df.copy()
df_shift['POS販売冊数'] = df_shift['POS販売冊数'] + SHIFT

df_log = df.copy()
df_log['POS販売冊数'] = np.log(df_log['POS販売冊数'] + math.e)

df_root_shift = df.copy()
df_root_shift['POS販売冊数'] = np.power(df_root_shift['POS販売冊数'] + SHIFT, 1/ROOT_N)
print("Complete!")

def estimate_tweedie_p(df, transform_name):
    stats = df.groupby('書名', observed=False)['POS販売冊数'].agg(['mean', 'var', 'count'])
    print(f"\n=== Statistics Summary ({transform_name}) ===")
    print(f"Mean: {stats['mean'].mean():.4f}")
    print(f"Median: {stats['mean'].median():.4f}")
    print(f"Variance: {stats['var'].mean():.4f}")
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
    plt.title(f'Mean-Variance Relationship (p={estimated_p:.2f})')
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
        label = f'Actual Distribution (log_add={params["log_add"]:.2f})'
    elif transform_type == 'root_shift':
        label = f'Actual Distribution (shift={params["shift"]}, root_n={params["root_n"]})'
    else:
        label = 'Actual Distribution'

    # Use appropriate bin width based on transform type
    if transform_type in ['original', 'shift']:
        # Integer data: use bins of width 1
        bins = np.arange(data.min(), data.max() + 2, 1)
    else:
        # Continuous data (log, root_shift): use finer bins
        bin_width = 0.05
        bins = np.arange(data.min(), data.max() + bin_width, bin_width)
    plt.hist(data, bins=bins, density=True, alpha=0.6, color='blue', label=label)

    x_range = np.linspace(data.min(), data.max(), 1000)
    from scipy.stats import norm, gamma, pareto, nbinom
    from tweedie import tweedie

    # Normal Distribution
    plt.plot(x_range, norm.pdf(x_range, mean_val, std_val), 'r-', linewidth=2, label=f'Normal Distribution\n(μ={mean_val:.3f}, σ={std_val:.3f})')

    # Tweedie Distribution
    if estimated_p < 2.0:
        tweedie_pdf = tweedie(p=estimated_p, mu=mean_val, phi=estimated_phi).pdf(x_range)
        plt.plot(x_range, tweedie_pdf, 'g-', linewidth=2, label=f'Tweedie Distribution\n(p={estimated_p:.3f}, φ={estimated_phi:.3f})')
    else:
        print(f"Tweedie distribution skipped (p={estimated_p:.3f} >= 2.0)")

    # Gamma Distribution
    # Estimate gamma parameters from mean and variance
    k_gamma = (mean_val ** 2) / (std_val ** 2)
    theta_gamma = (std_val ** 2) / mean_val
    gamma_pdf = gamma.pdf(x_range, k_gamma, scale=theta_gamma)
    plt.plot(x_range, gamma_pdf, 'b-', linewidth=2, label=f'Gamma Distribution\n(k={k_gamma:.3f}, θ={theta_gamma:.3f})')

    # Pareto Distribution
    # Fit pareto distribution to data with fixed location and scale
    # No need to filter data > 0 since shift transformation ensures all data is positive
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
        pass  # Skip Pareto distribution if fitting fails

    # ==========================================
    # Hurdle Model (Zero-Truncated Negative Binomial)
    # "0"と"正の値"を完全に分けてモデル化する手法
    # ==========================================
    # 1. パラメータ推定
    # ゼロの割合 (データの実測値をそのまま信頼する)
    pi_est = (data == 0).mean()

    # 0以外のデータから n, p を推定
    data_nz = data[data > 0]
    if len(data_nz) > 0:
        mean_nz = data_nz.mean()
        var_nz = data_nz.var()

        # 負の二項分布の条件チェック
        if var_nz > mean_nz:
            p_est = mean_nz / var_nz
            n_est = (mean_nz**2) / (var_nz - mean_nz)
        else:
            p_est = 0.99
            n_est = mean_nz * p_est / (1 - p_est)
    else:
        p_est, n_est = 0.5, 1.0

    # 2. 確率密度(PMF)の計算
    x_int = np.arange(0, int(x_range.max()) + 1)

    # 通常の負の二項分布を計算
    nb_pmf = nbinom.pmf(x_int, n_est, p_est)

    # 【重要】0の確率(P_nb_zero)を計算して、0より大きい部分を正規化する
    prob_nb_zero = nbinom.pmf(0, n_est, p_est)

    # Hurdle Modelの確率計算
    hurdle_pmf = np.zeros_like(nb_pmf)

    # x=0 の確率は、データの0の割合(pi_est)そのものを使う
    hurdle_pmf[0] = pi_est

    # x > 0 の確率は、NB分布の正の部分を (1 - pi_est) に収まるように引き伸ばす
    # P(k | k>0) = P_nb(k) / (1 - P_nb(0))
    if prob_nb_zero < 1.0:
        scale_factor = (1 - pi_est) / (1 - prob_nb_zero)
        hurdle_pmf[1:] = nb_pmf[1:] * scale_factor

    # 3. 描画 (オレンジ色の破線)
    plt.plot(x_int, hurdle_pmf, 'orange', linestyle='--', marker='o', markersize=3, alpha=0.7,
             label=f'Hurdle Model (NB)\n($\\pi$={pi_est:.2f}, n={n_est:.2f}, p={p_est:.2f})')
    # ==========================================

    # ==========================================
    # Model Comparison: Log-Likelihood, AIC, BIC
    # ==========================================
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

    # Hurdle Model
    data_int = data.astype(int)
    ll_hurdle = 0
    for val in data_int:
        if val < len(hurdle_pmf):
            prob = hurdle_pmf[val]
            if prob > 0:
                ll_hurdle += np.log(prob)
            else:
                ll_hurdle += -np.inf
    k_hurdle = 3
    aic_hurdle = -2 * ll_hurdle + 2 * k_hurdle
    bic_hurdle = -2 * ll_hurdle + k_hurdle * np.log(n_data)
    print(f"Hurdle (NB) - AIC: {aic_hurdle:.2f}, BIC: {bic_hurdle:.2f}")
    # ==========================================

    if transform_type == 'original':
        xlabel = 'POS Sales'
    elif transform_type == 'shift':
        xlabel = f'POS Sales (shift by {params["shift"]})'
    elif transform_type == 'log':
        xlabel = f'POS Sales (log, add {params["log_add"]:.2f})'
    elif transform_type == 'root_shift':
        xlabel = f'POS Sales (shift {params["shift"]}, {params["root_n"]}-th root)'
    else:
        xlabel = 'POS Sales'

    plt.xlabel(xlabel)
    plt.ylabel('Density')
    plt.title('Distribution of POS vs Normal and Tweedie Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)

    """
    """
    # ========== ZOOM LEFT SIDE (DELETE THIS SECTION TO RESTORE FULL VIEW) ==========
    if transform_type in ['original', 'shift', 'root_shift']:
        max_x = data.max()
        zoom_limit = max_x * 0.01  # Show only left 10% of the range
        plt.xlim(0, zoom_limit)
    # ========== END ZOOM ==========

    output_dir = f'figure/{transform_type}'
    os.makedirs(output_dir, exist_ok=True)

    output_path = f'{output_dir}/{transform_type}_distribution_comparison.png'
    plt.savefig(output_path)
    print(f"Saved: {output_path}")

plot_plot(df_original, transform_type='original')
plot_plot(df_shift, transform_type='shift', shift=SHIFT)
plot_plot(df_log, transform_type='log', log_add=math.e)
plot_plot(df_root_shift, transform_type='root_shift', shift=SHIFT, root_n=ROOT_N)
"""
class Tweedie(Distribution):
    arg_constraints = {}

    def __init__(self, mu, rho, validate_args=None):
        if mu.dim() > 2 and mu.shape[-1] == 1:
            mu = mu.squeeze(-1)
        self.mu = mu
        self.rho = rho
        super().__init__(batch_shape=mu.shape, validate_args=validate_args)

    def log_prob(self, value):
        loss = F.tweedie_loss(self.mu, value, p=self.rho, reduction='none')
        return -loss

    def sample(self, sample_shape=torch.Size()):
        extended_shape = sample_shape + self.mu.shape
        mu_expanded = self.mu.expand(extended_shape)

        phi = 1.0
        p = self.rho
        mu_safe = torch.clamp(mu_expanded, min=1e-8)

        lambda_val = (mu_safe ** (2 - p)) / (phi * (2 - p))
        alpha = (2 - p) / (p - 1)
        beta = phi * (p - 1) * (mu_safe ** (p - 1))

        n_samples = torch.poisson(lambda_val)

        gamma_samples = torch.zeros_like(n_samples)
        mask = n_samples > 0

        if mask.any():
            valid_n = n_samples[mask]
            m = torch.distributions.Gamma(concentration=valid_n * alpha, rate=1.0/beta[mask])
            gamma_samples[mask] = m.sample()
        return gamma_samples

    @property
    def mean(self):
        return self.mu

class LightGBMTweedieOutput(Output):
    args_dim = {"mu": 1}

    def __init__(self, rho: float = 1.5):
        assert 1.0 < rho < 2.0, "rho (p) must be between 1.0 and 2.0"
        self.rho = rho

    def domain_map(self, mu):
        # 平均値は正である必要がある
        return F.softplus(mu)

    def distribution(self, distr_args, loc=None, scale=None):
        return LightGBMTweedie(distr_args, rho=self.rho)

    def loss(self, target, distr_args, loc=None, scale=None):
        distr = self.distribution(distr_args, loc=loc, scale=scale)
        return -distr.log_prob(target)

    @property
    def event_shape(self):
        return ()

    @property
    def forecast_generator(self):
        # サンプリングベースの予測生成器を指定
        return SampleForecastGenerator()
"""