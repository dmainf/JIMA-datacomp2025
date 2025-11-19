import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson, gamma

# --- Configuration Parameters ---
MU = 10.0  # Mean E[X]
PHI = 0.2  # Dispersion Parameter φ
N_SAMPLES = 10000  # Number of samples for simulation
P_VALUES = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]  # Tweedie power parameter values

def tweedie_sampler_full(p, mu, phi, n_samples):
    """
    Samples from the Tweedie family covering p=1 (Poisson), 1 < p < 2 (Compound), and p=2 (Gamma).
    """
    if p == 1.0:
        # p=1.0: Poisson Distribution (Discrete count data)
        # Var(X) = mu (phi assumed to be 1 for standard Poisson)
        return poisson.rvs(mu, size=n_samples)
    
    elif p == 2.0:
        # p=2.0: Gamma Distribution (Continuous positive values)
        # Var(X) = phi * mu^2. Parameters derived from mean/variance relationship.
        alpha_k = 1 / phi
        beta_theta = mu * phi
        return gamma.rvs(a=alpha_k, scale=beta_theta, size=n_samples)
    
    elif 1.0 < p < 2.0:
        # 1 < p < 2: Compound Poisson-Gamma Distribution (Custom logic required)
        
        # 1. Calculate Poisson mean (lambda) for the number of events
        lambda_p = (mu**(2 - p)) / ((2 - p) * phi)
        
        # 2. Calculate Gamma shape (alpha_k) and scale (beta_theta) for event size
        alpha_k = (2 - p) / (p - 1)
        beta_theta = phi * (p - 1) * (mu**(p - 1))
        
        # 3. Perform Sampling
        n_events = poisson.rvs(lambda_p, size=n_samples) # Number of gamma events
        samples = np.zeros(n_samples)
        
        for i in range(n_samples):
            if n_events[i] > 0:
                # Summing n_events[i] independent Gamma random variables
                samples[i] = np.sum(gamma.rvs(a=alpha_k, scale=beta_theta, size=n_events[i]))
        return samples

    else:
        raise ValueError("Supported range for p is [1.0, 2.0].")

# --- Execute Plotting (2 Rows x 3 Columns) ---
fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True, sharey=True)
axes = axes.flatten()

for i, p in enumerate(P_VALUES):
    try:
        # Generate data
        data = tweedie_sampler_full(p, MU, PHI, N_SAMPLES)
        
        ax = axes[i]
        
        # Plot histogram
        ax.hist(data, bins=60, density=True, color='purple', alpha=0.6, edgecolor='gray', linewidth=0.5)
        
        # Determine P(X=0)
        if p == 1.0:
             # Poisson PMF at 0
             zero_prob = poisson.pmf(0, MU)
        elif 1.0 < p < 2.0:
             # Count sampled zeros
             zero_count = np.sum(data == 0)
             zero_prob = zero_count / N_SAMPLES
        else: # p=2.0
             # Gamma is continuous, P(X=0) = 0
             zero_prob = 0.0
        
        # P(X=0) annotation
        if zero_prob > 0.001:
            ax.text(0.5, 0.85, f'P(X=0) $\\approx$ {zero_prob:.3f}', 
                    transform=ax.transAxes, fontsize=12, color='red', weight='bold')

        ax.set_title(f'Tweedie Distribution (p={p:.1f})')
        ax.set_xlabel('X (Value)')
        ax.set_ylabel('Density')

    except ValueError as e:
        ax = axes[i]
        ax.text(0.5, 0.5, f'Error for p={p}: {e}', transform=ax.transAxes, ha='center')

plt.suptitle(f'Tweedie Distribution Shape Change (μ={MU}, $\\phi$={PHI})', fontsize=18)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()