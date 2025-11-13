
'''

q: charlotte vogt

'''



import numpy as np
import pandas as pd
import os
import matplotlib as mpl
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
import svgutils.transform as sg
from svgutils.compose import Unit
from cairosvg import svg2png
import random

mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'Arial'

input_spectra_path = '/Users/charlottevogt/Documents/Manuscripts/Manifesto Experimental - Square vs Saw/Simulation/Tiered DOE Framework/Sample_Spectra_SixSpecies_Biochemical.csv'
output_base = '/Users/charlottevogt/Documents/Manuscripts/Manifesto Experimental - Square vs Saw/Simulation/Tiered DOE Framework/01062025 - Tiered  LHS'
os.makedirs(output_base, exist_ok=True)
lhs_csv_path = os.path.join(output_base, "LHS_Design_Space.csv")

def sigmoid_threshold(x, y, k=5, theta=0.5): return 1 / (1 + np.exp(-k * (x - theta)))
def nonlinear_saturating(x, y): return np.tanh(x - y)
def nonlinear_stochastic(x, y): return np.random.choice([0, 1], p=[0.98, 0.02]) * (x + y)

nonlinear_map = {
    'sigmoid_threshold': lambda x, y: sigmoid_threshold(x, y),
    "saturating": nonlinear_saturating,
    "stochastic": nonlinear_stochastic,
    None: lambda x, y: 0
}

def get_modulation(name, t, period, mode='full'):
    if mode == '1pulse':
        t_pulse = t[t <= period]
    else:
        t_pulse = t

    if name == "square":
        factor = 10000
        t_fine = np.linspace(t[0], t[-1], len(t) * factor)
        
        square_fine = ((t_fine % period) < (period / 2)).astype(float)

        interp_func = interp1d(t_fine, square_fine, kind='nearest', bounds_error=False, fill_value="extrapolate")
        return interp_func(t)

    elif name == "sine":
        return 0.5 * (1 + np.sin(2 * np.pi * t / period))

    elif name == "sawtooth":
        return (t % period) / period
    
    elif name == "constant":
        mod = np.ones_like(t_pulse)
    else:
        raise ValueError("Unknown modulation type")

    if mode == '1pulse':
        full_mod = np.zeros_like(t)
        full_mod[:len(mod)] = mod
        return full_mod

    return mod

def load_component_spectra(path, n_components):
    df = pd.read_csv(path)
    energy = df.iloc[:, 0].values
    spectra = [df.iloc[:, i].values for i in range(1, 1 + n_components)]
    return [(energy, s) for s in spectra]

MAX_VALUE = 1e3

def solve_response(t, tau, amps, alpha, beta, modulation, nl_matrix, offset,
                   regime="on_on", response_types=None):
    n_comp = len(tau)
    dt = t[1] - t[0]
    n_t = len(t)

    if response_types is None:
        response_types = ["linear"] * n_comp

    R_hat = np.zeros((n_comp, n_t))
    dR_hat = np.zeros((n_comp, n_t))
    L = np.zeros_like(R_hat)
    N = np.zeros_like(R_hat)
    E = np.zeros_like(R_hat)
    F = np.zeros_like(R_hat)

    for i in range(1, n_t):
        S_t = modulation[i - 1]
        for k in range(n_comp):
            R_k_prev = R_hat[k, i - 1]
            activation = get_activation_mask(S_t, regime, k)

            if response_types[k] == "linear":
                Fi_Rk = R_k_prev
            elif response_types[k] == "saturating":
                Fi_Rk = np.tanh(R_k_prev)
            else:
                raise ValueError(f"Unknown response type: {response_types[k]}")

            F[k, i] = Fi_Rk

            drive_term = activation * (S_t - Fi_Rk)
            E_term = (1.0 / tau[k]) * drive_term

            L_term = sum(alpha[k, j] * (R_hat[j, i - 1] - R_k_prev)
                         for j in range(n_comp) if j != k)

            N_term = sum(beta[k, j] * nl_matrix[k][j](R_hat[j, i - 1], R_k_prev)
                         for j in range(n_comp)
                         if j != k and nl_matrix[k][j] is not None)

            dR = E_term + L_term + N_term
            dR_hat[k, i] = dR
            R_hat[k, i] = R_k_prev + dt * dR

            E[k, i] = E_term
            L[k, i] = L_term
            N[k, i] = N_term

    R = np.array([amps[k] * R_hat[k] for k in range(n_comp)]).T
    dR = dR_hat.T
    E = E.T
    L = L.T
    N = N.T
    F = F.T

    return R, L, N, E, dR, F

def get_nonlinear_function(name):
    def sigmoid_threshold(x, y, k=5, theta=0.5):
        return 1 / (1 + np.exp(-k * (x - theta)))

    if name == "sigmoid_threshold":
        return lambda x, y: sigmoid_threshold(x, y)
    elif name == "saturating":
        return lambda x, y: np.tanh(x - y)
    elif name == "stochastic":
        return lambda x, y: np.random.choice([0, 1], p=[0.98, 0.02]) * (x + y)
    else:
        raise ValueError(f"Unknown nonlinear function: {name}")

def nonlinear_input(S, kind):
    if kind == "linear":
        return S
    elif kind == "saturating":
        return np.tanh(S)
    elif kind == "threshold":
        threshold = 0.8
        sharpness = 20
        return np.tanh(sharpness * (S - threshold))
    else:
        raise ValueError(f"Unknown response type: {kind}")

def get_activation_mask(S, regime, component_index):
    if regime == "off_off":
        return 1.0 if (component_index == 0 and S > 0.9) else 0.0
    elif regime == "off_on":
        return 1.0 if S > 0.9 else 0.0
    elif regime == "on_on":
        return 1.0
    else:
        raise ValueError(f"Unknown regime: {regime}")

def generate_spectral_matrix(R, component_spectra):
    energy = component_spectra[0][0]
    matrix = np.zeros((len(R), len(energy)))
    for i, (_, s) in enumerate(component_spectra):
        matrix += np.outer(R[:, i], s)
    return energy, matrix

colors_raw = ["#355C7D", "#F8B739", "#E38190"]
custom_smooth_cmap = LinearSegmentedColormap.from_list("smooth_cmap", colors_raw, N=256)
cmap = custom_smooth_cmap
colors = cmap(np.linspace(0, 1, 6))

def create_three_panel_figure_square_panels(panel_size=5, spacing=1.5):
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    total_width = panel_size * 3 + spacing * 2
    total_height = panel_size

    fig = plt.figure(figsize=(total_width, total_height), constrained_layout=False)
    spec = GridSpec(nrows=1, ncols=3, width_ratios=[1, 1, 1], figure=fig,
                    wspace=spacing / panel_size)

    axs = [fig.add_subplot(spec[0, i]) for i in range(3)]
    for ax in axs:
        ax.tick_params(direction='out')

    return fig, axs

def plot_component_spectra(path, n_components=5, panel_size=5, output_path=None):
    df = pd.read_csv(path)
    energy = df.iloc[:, 0].values
    spectra = [df.iloc[:, i].values for i in range(1, 1 + n_components)]

    fig, ax = plt.subplots(figsize=(panel_size, panel_size))

    for i, spectrum in enumerate(spectra):
        ax.plot(energy, spectrum, label=f'Component {i+1}', color=colors[i])

    ax.set_xlabel("Wavenumber ($\\mathrm{{cm}}^{{-1}}$)")
    ax.set_ylabel("Absorbance (a.u.)")
    ax.set_title("Component Spectra")
    ax.set_xlim(4000, 650)
    ax.legend()
    ax.tick_params(direction='out')

    if output_path:
        plt.savefig(output_path + ".png", dpi=300)
        plt.savefig(output_path + ".svg", format='svg')
    plt.show()
    plt.close()

plot_component_spectra(
    path=input_spectra_path,
    n_components=5,
    output_path=os.path.join(output_base, "component_spectra")
)

def save_outputs(tag, t, R, L, N, E, dR, energy, spectral_matrix, base_dir, modulation, Fi):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import os
    import pandas as pd
    import numpy as np
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    folder = os.path.join(base_dir, tag)
    os.makedirs(folder, exist_ok=True)

    df = pd.DataFrame({"Time (s)": t, "S(t)": modulation})
    for i in range(R.shape[1]):
        df[f"R{i+1}"] = R[:, i]
        df[f"dR{i+1}/dt"] = dR[:, i]
        df[f"τ_contrib_{i+1}"] = E[:, i]
        df[f"L_contrib_{i+1}"] = L[:, i]
        df[f"N_contrib_{i+1}"] = N[:, i]
        df[f"Fi_{i+1}"] = Fi[:, i]
    df.to_csv(os.path.join(folder, "responses.csv"), index=False)

    df_spec = pd.DataFrame(spectral_matrix.T, index=energy)
    df_spec.to_csv(os.path.join(folder, "spectral_matrix.csv"))

    panel_size = 5
    fig, ax = plt.subplots(figsize=(panel_size, panel_size))
    aspect_ratio = (t[-1] - t[0]) / abs(energy[-1] - energy[0])

    cax = ax.imshow(spectral_matrix.T, aspect=aspect_ratio, origin='lower',
                    extent=[t[0], t[-1], energy[0], energy[-1]], cmap='viridis')

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Wavenumber ($\\mathrm{cm}^{-1}$)")
    ax.set_title("Simulated Spectral Matrix")
    ax.invert_yaxis()

    cb_ax = inset_axes(ax, width="5%", height="25%", loc='upper right',
                       bbox_to_anchor=(-0.11, -0.02, 1, 1),
                       bbox_transform=ax.transAxes, borderpad=0)
    cb = fig.colorbar(cax, cax=cb_ax)
    cb.ax.tick_params(labelsize=8)
    cb.ax.yaxis.set_ticks([0, 1])
    cb.set_label("Intensity", fontsize=9)

    save_base = os.path.join(folder, "spectral_heatmap")
    plt.savefig(save_base + ".png", dpi=300)
    plt.savefig(save_base + ".svg", format='svg')
    plt.close()

    fig_L, ax_L = plt.subplots(figsize=(6, 4))
    for i in range(L.shape[1]):
        ax_L.plot(t, L[:, i], label=f'L contrib R{i+1}')
    ax_L.set_xlabel("Time (s)")
    ax_L.set_ylabel("Linear Contribution")
    ax_L.set_title("Linear Coupling Contribution per Component")
    ax_L.legend()
    fig_L.tight_layout()
    fig_L.savefig(os.path.join(folder, "linear_contributions.png"), dpi=300)
    fig_L.savefig(os.path.join(folder, "linear_contributions.svg"), format='svg')
    plt.close(fig_L)

    fig_N, ax_N = plt.subplots(figsize=(6, 4))
    for i in range(N.shape[1]):
        ax_N.plot(t, N[:, i], label=f'N contrib R{i+1}')
    ax_N.set_xlabel("Time (s)")
    ax_N.set_ylabel("Nonlinear Contribution")
    ax_N.set_title("Nonlinear Coupling Contribution per Component")
    ax_N.legend()
    fig_N.tight_layout()
    fig_N.savefig(os.path.join(folder, "nonlinear_contributions.png"), dpi=300)
    fig_N.savefig(os.path.join(folder, "nonlinear_contributions.svg"), format='svg')
    plt.close(fig_N)

    fig_Fi, ax_Fi = plt.subplots(figsize=(6, 4))
    for i in range(Fi.shape[1]):
        ax_Fi.plot(t, Fi[:, i], label=f'F_i(R{i+1})')
    ax_Fi.set_xlabel("Time (s)")
    ax_Fi.set_ylabel("F_i(R)")
    ax_Fi.set_title("Intrinsic Nonlinear Response F_i(R) per Component")
    ax_Fi.legend()
    fig_Fi.tight_layout()
    fig_Fi.savefig(os.path.join(folder, "Fi_response.png"), dpi=300)
    fig_Fi.savefig(os.path.join(folder, "Fi_response.svg"), format='svg')
    plt.close(fig_Fi)

    def create_three_panel_figure_square_panels(panel_size=5, spacing=1.5):
        fig = plt.figure(figsize=(panel_size * 3 + spacing * 2, panel_size), constrained_layout=False)
        spec = gridspec.GridSpec(nrows=1, ncols=3, width_ratios=[1, 1, 1], figure=fig,
                                 wspace=spacing / panel_size)
        axs = [fig.add_subplot(spec[0, i]) for i in range(3)]
        for ax in axs:
            ax.tick_params(direction='out')
        return fig, axs

    fig, axs = create_three_panel_figure_square_panels(panel_size=5)

    axs[0].plot(t, modulation, color="black", label="S(t)")
    axs[0].set_ylabel("S(t)")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_title("Input Modulation")
    axs[0].legend()
    axs[0].set_xlim(t[0], t[-1])

    for i in range(R.shape[1]):
        axs[1].plot(t[1:], dR[1:, i], label=f'dR$_{{{i+1}}}$/dt', color=colors[i])
    axs[1].set_ylabel("dR$_i$/dt")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_title("Dynamic Response Rates")
    axs[1].legend()
    axs[1].set_xlim(t[1], t[-1])
    
    for i in range(R.shape[1]):
        axs[2].plot(t[1:], R[1:, i], label=f'R$_{{{i+1}}}$', color=colors[i])
    axs[2].set_ylabel("R$_i$(t)")
    axs[2].set_xlabel("Time (s)")
    axs[2].set_title("Component Responses")
    axs[2].legend()
    axs[2].set_xlim(t[0], t[-1])

    save_base = os.path.join(folder, "full_response_decomposition")
    plt.savefig(save_base + ".png", dpi=300)
    plt.savefig(save_base + ".svg", format='svg')
    plt.close()
    print(f"Saved both spectral and response panels at: {folder}")

import numpy as np
import pandas as pd
import os
from scipy.stats import qmc

from itertools import product
from scipy.stats import qmc
import numpy as np

def generate_stratified_lhs_design(n_samples_per_block=5, seed=42, save_path=None):
    from scipy.stats import qmc
    import pandas as pd
    import numpy as np

    rng = np.random.default_rng(seed)

    modulation_types = ['square', 'sine', 'sawtooth', 'constant']
    spectral_overlap_cases = ['full', 'partial', 'none']
    nonlinearity_options = ['sigmoid_threshold', 'saturating', 'stochastic']
    activation_masks = ['00', '01', '11']
    tau_range = (1.0, 10.0)
    alpha_range = (0.0, 0.5)
    beta_range = (0.0, 0.5)
    modulation_period_range = (1, 100)
    noise_range = (0.0, 0.5)
    amp_sets = [
        [1.0, 1.0, 1.0],
        [1.0, 0.5, 0.2],
        [1.0, 1.0, 1.0, 0.8],
        [1.0, 0.8, 0.6, 0.4, 0.2]
    ]

    offset_map = {
        3: [[0.0, 0.0, 0.1], None],
        4: [[0.0, 0.0, 0.1, 0.1], None],
        5: [[0.0, 0.0, 0.1, 0.1, 0.1], None]
    }

    coupled_map = {
        3: [[0], [0, 1], [0, 1, 2]],
        4: [[0, 1, 2], [0, 2, 3]],
        5: [[1, 2, 3, 4]]
    }

    pulse_modes = [1, 'full']
    Fi_options = ['linear', 'saturating']

    all_rows = []

    for mod_type in modulation_types:
        for overlap_case in spectral_overlap_cases:
            sampler = qmc.LatinHypercube(d=6, seed=rng.integers(0, 1e6))
            lhs_samples = sampler.random(n=n_samples_per_block)

            for s in lhs_samples:
                alpha = alpha_range[0] + s[0] * (alpha_range[1] - alpha_range[0])
                beta = beta_range[0] + s[1] * (beta_range[1] - beta_range[0])
                mod_period = modulation_period_range[0] + s[2] * (modulation_period_range[1] - modulation_period_range[0])
                noise = noise_range[0] + s[4] * (noise_range[1] - noise_range[0])
                nonlinearity = nonlinearity_options[int(s[5] * len(nonlinearity_options))]

                amps = amp_sets[rng.integers(0, len(amp_sets))]
                n_components = len(amps)

                tau = list(rng.uniform(tau_range[0], tau_range[1], size=n_components))

                if n_components not in offset_map or n_components not in coupled_map:
                    continue
                offset = offset_map[n_components][rng.integers(0, len(offset_map[n_components]))]
                coupled = coupled_map[n_components][rng.integers(0, len(coupled_map[n_components]))]

                pulse_mode = pulse_modes[rng.integers(0, len(pulse_modes))]
                act_mask = activation_masks[rng.integers(0, len(activation_masks))]

                Fi_list = [Fi_options[rng.integers(0, len(Fi_options))] for _ in range(n_components)]
                Fi_str = "-".join(Fi_list)

                tau_str = "_".join([f"{v:.1f}" for v in tau])

                tag = (
                    f"LHS_tau({tau_str})_alpha{round(alpha, 2)}_beta{round(beta, 2)}_"
                    f"{nonlinearity}_{mod_type}{round(mod_period)}s_{pulse_mode}pulse_noise{round(noise, 3)}_"
                    f"amps{str(tuple(amps))}_offset{offset is not None}_"
                    f"coupled{''.join(map(str, coupled))}_n{n_components}_"
                    f"overlap_{overlap_case}_act_{act_mask}_Fi({Fi_str})"
                )

                all_rows.append({
                    "tag": tag,
                    "taus": tau,
                    "alpha": alpha,
                    "beta": beta,
                    "nonlinearity": nonlinearity,
                    "mod_type": mod_type,
                    "mod_period": mod_period,
                    "pulse_mode": pulse_mode,
                    "noise": noise,
                    "amplitudes": amps,
                    "offset_mask": offset,
                    "coupled": coupled,
                    "n_components": n_components,
                    "spectral_overlap": overlap_case,
                    "activation_mask": act_mask,
                    "Fi_per_component": Fi_list
                })

    df = pd.DataFrame(all_rows)
    if save_path:
        df.to_csv(save_path, index=False)

    print(f"Stratified LHS design generated with {len(df)} samples.")
    return df

def add_uncoupled_complements(design_df):
    additional_rows = []
    existing_tags = set(design_df['tag'])

    for idx, row in design_df.iterrows():
        if row['coupled'] == [0]:
            continue

        new_row = row.copy()
        new_row['coupled'] = [0]

        new_tag = row['tag']
        if "coupled" in new_tag:
            new_tag = re.sub(r"coupled\d+", "coupled0", new_tag)
        else:
            new_tag += "_coupled0"
        new_row['tag'] = new_tag

        if new_tag not in existing_tags:
            additional_rows.append(new_row)
            existing_tags.add(new_tag)

    if additional_rows:
        print(f"Added {len(additional_rows)} uncoupled simulations.")
        design_df = pd.concat([design_df, pd.DataFrame(additional_rows)], ignore_index=True)
    else:
        print("No additional uncoupled simulations were needed.")
    
    return design_df

def run_extended_batch_with_design_df(design_df):
    print("Running DOE on", len(design_df), "designs...")
    for idx, row in design_df.iterrows():
        taus = list(row["taus"]) + [row["taus"][-1]] * (row["n_components"] - len(row["taus"]))
        if any(t <= 0 for t in taus):
            print(f"Skipping {row['tag']}: invalid tau values.")
            continue

        amps = row["amplitudes"]
        n = row["n_components"]
        offset = np.array(row["offset_mask"]) if row["offset_mask"] is not None else None
        alpha_val = row["alpha"]
        beta_val = row["beta"]
        mod_type = row["mod_type"]
        mod_period = row["mod_period"]
        pulse_mode = row["pulse_mode"]
        noise = row["noise"]
        tag = row["tag"]
        coupled = row["coupled"]
        nl_base = row["nonlinearity"]

        response_types = ["saturating" if "saturating" in nl_base else "linear"] * n

        total_time = mod_period * 10 * 2 if pulse_mode == "full" else mod_period * int(pulse_mode) * 2
        t = np.arange(0, 100, 1.0)
        mod_signal = get_modulation(mod_type, t, mod_period, mode=str(pulse_mode))

        alpha = np.zeros((n, n))
        beta = np.zeros((n, n))
        nl_matrix = [[None for _ in range(n)] for _ in range(n)]
        for j in coupled:
            for k in coupled:
                if j != k:
                    alpha[j, k] = alpha_val
                    beta[j, k] = beta_val
                    nl_matrix[j][k] = get_nonlinear_function(nl_base)

        component_spectra = load_component_spectra(input_spectra_path, n)

        R, L, N, E, dR, Fi = solve_response(
            t, taus, amps, alpha, beta, mod_signal, nl_matrix, offset,
            regime="on_on", response_types=response_types
        )

        energy, spec_matrix = generate_spectral_matrix(R, component_spectra)

        if noise > 0:
            spec_matrix += np.random.normal(0, noise, size=spec_matrix.shape)

        save_outputs(tag, t, R, L, N, E, dR, energy, spec_matrix, output_base, mod_signal, Fi=Fi)

    print("DOE run complete.")

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns

def visualize_and_filter_lhs(df, output_base, method='pca', save_prefix='lhs_sample_space'):
    feature_cols = ["taus", "alpha", "beta", "nonlinearity", "mod_type", "mod_period",
                    "pulse_mode", "noise", "amplitudes", "offset_mask", "coupled", "n_components"]

    def flatten_row(row):
        return (
            list(row["taus"]) +
            [row["alpha"], row["beta"], row["mod_period"], float(row["noise"]), row["n_components"]] +
            list(row["amplitudes"]) +
            ([0] * (5 - len(row["amplitudes"]))) +
            [hash(row["nonlinearity"]) % 10, hash(row["mod_type"]) % 10, hash(str(row["pulse_mode"])) % 10] +
            ([sum(row["offset_mask"])] if row["offset_mask"] is not None else [0]) +
            [sum(row["coupled"])]
        )

    flattened = np.array([flatten_row(r) for _, r in df.iterrows()])
    
    _, unique_indices = np.unique(flattened, axis=0, return_index=True)
    df_unique = df.iloc[sorted(unique_indices)].reset_index(drop=True)

    print(f"Filtered {len(df) - len(df_unique)} redundant entries from LHS design space.")
    
    filtered_path = os.path.join(output_base, "LHS_Design_Space_filtered.csv")
    df_unique.to_csv(filtered_path, index=False)
    print("Filtered LHS saved to:", filtered_path)

    if method == 'pca':
        reducer = PCA(n_components=2)
    elif method == 'tsne':
        reducer = TSNE(n_components=2, perplexity=30, random_state=42)
    else:
        raise ValueError("method must be 'pca' or 'tsne'")
    
    coords = reducer.fit_transform(flattened)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=coords[:, 0], y=coords[:, 1])
    plt.title(f"LHS Sample Space ({method.upper()})")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(os.path.join(output_base, f"{save_prefix}_{method}.png"), dpi=300)
    plt.savefig(os.path.join(output_base, f"{save_prefix}_{method}.svg"))
    plt.close()

    return df_unique

design_df = generate_stratified_lhs_design(n_samples_per_block=5, seed=42)

import re
design_df = add_uncoupled_complements(design_df)

design_df.to_csv(lhs_csv_path, index=False)
print(f"Combined design (PC1 + PC2) saved to: {lhs_csv_path}")

run_extended_batch_with_design_df(design_df)

from sklearn.decomposition import PCA

def run_pca_on_all_spectral_matrices(base_dir, n_components=5):
    for folder_name in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder_name)
        matrix_path = os.path.join(folder_path, "spectral_matrix.csv")

        if not os.path.isfile(matrix_path):
            continue

        df = pd.read_csv(matrix_path, index_col=0)
        spectral_matrix = df.values.T
        times = np.array([float(c) for c in df.columns])
        wavenumbers = df.index.values.astype(float)

        pca = PCA(n_components=n_components)
        scores = pca.fit_transform(spectral_matrix)
        loadings = pca.components_
        explained = pca.explained_variance_ratio_
        cve = np.cumsum(explained)

        fig, axs = create_three_panel_figure_square_panels(panel_size=5)

        for i in range(n_components):
            axs[0].plot(times, scores[:, i], label=f'PC{i+1}', color=colors[i])
        axs[0].set_xlabel("Time (s)")
        axs[0].set_ylabel("Score")
        axs[0].set_xlim(times[0], times[-1])
        axs[0].set_title("PCA Scores vs. Time")
        axs[0].legend()
        axs[0].tick_params(direction='out')

        offset_step = 0.12
        for i in range(n_components):
            offset = offset_step * (n_components - i - 1)
            axs[1].plot(wavenumbers, loadings[i] + offset, label=f'PC{i+1}', color=colors[i])
        axs[1].set_xlabel("Wavenumber ($\\mathrm{cm}^{-1}$)")
        axs[1].set_xlim(4000, 650)
        axs[1].set_yticks([])
        axs[1].set_title("PCA Loadings (Offset)")
        axs[1].legend(loc="upper right")
        axs[1].tick_params(direction='out')

        scalebar_value = 0.05
        spacing = 0.03
        center_x_frac = 0.95
        center_y_frac = 0.07

        axs[1].plot([center_x_frac, center_x_frac],
                    [center_y_frac - scalebar_value / 2, center_y_frac + scalebar_value / 2],
                    transform=axs[1].transAxes, color="black", lw=1.5, clip_on=False)
        axs[1].text(center_x_frac - spacing, center_y_frac,
                    f"{scalebar_value:.2f} arb. u.",
                    transform=axs[1].transAxes, va='center', ha='right', fontsize=10)

        ax1 = axs[2]
        ax2 = ax1.twinx()
        pc_indices = np.arange(1, n_components + 1)
        ax1.plot(pc_indices, cve, marker='o', color=colors[0])
        ax1.set_ylim(0.95, 1.005)
        ax1.set_ylabel("Cumulative Variance")
        ax1.set_xlabel("Principal Component")
        ax1.set_xticks(pc_indices)
        ax1.set_title("PCA Variance Explained")
        ax1.tick_params(direction='out')

        ax2.bar(pc_indices[1:], explained[1:], width=0.4, alpha=0.3, color=colors[1:n_components])
        ax2.set_ylim(0, max(explained[1:]) * 1.2)
        ax2.set_ylabel("Individual PC Variance (PC2–5)")
        ax2.tick_params(direction='out')
        
        score_df = pd.DataFrame(scores, columns=[f"PC{i+1}" for i in range(n_components)])
        score_df.insert(0, "Time (s)", times)
        score_df.to_csv(os.path.join(folder_path, "pca_scores.csv"), index=False)
        
        cve_df = pd.DataFrame({
            "Principal Component": [f"PC{i+1}" for i in range(n_components)],
            "Explained Variance": explained,
            "Cumulative Variance": cve
        })
        cve_df.to_csv(os.path.join(folder_path, "pca_variance.csv"), index=False)

        save_base = os.path.join(folder_path, "pca_combined")
        plt.savefig(save_base + ".png", dpi=300)
        plt.savefig(save_base + ".svg", format='svg')
        plt.close()
        print(f"PCA completed and saved for: {folder_name}")

run_pca_on_all_spectral_matrices(output_base)

import os
import svgutils.transform as sg

def combine_svgs_vertically(folder_path, top_file="full_response_decomposition.svg",
                            bottom_file="pca_combined.svg", output_name="combined_panel.svg",
                            labels=("A", "B", "C", "D", "E", "F"),
                            label_positions=((10, 20), (180, 20), (350, 20), (10, 240), (180, 240), (350, 240))):
    top_svg_path = os.path.join(folder_path, top_file)
    bottom_svg_path = os.path.join(folder_path, bottom_file)
    output_path = os.path.join(folder_path, output_name)

    try:
        top_fig = sg.fromfile(top_svg_path)
        bottom_fig = sg.fromfile(bottom_svg_path)

        width = float(top_fig.get_size()[0].replace("pt", ""))
        height_top = float(top_fig.get_size()[1].replace("pt", ""))
        height_bottom = float(bottom_fig.get_size()[1].replace("pt", ""))
        total_height = height_top + height_bottom

        combined = sg.SVGFigure(f"{width}px", f"{total_height}px")
        combined.set_size((f"{width}px", f"{total_height}px"))

        top_panel = top_fig.getroot()
        bottom_panel = bottom_fig.getroot()
        bottom_panel.moveto(0, height_top)

        combined.append([top_panel, bottom_panel])

        for i, (label, (x, y)) in enumerate(zip(labels, label_positions)):
            text = sg.TextElement(x, y, label, size=25, weight="bold", font="Arial")
            combined.append([text])

        combined.save(output_path)
        print(f"Combined SVG saved at: {output_path}")
        
        scale = 6
        svg2png(
            url=output_path,
            write_to=output_path.replace(".svg", ".png"),
            output_width=int(width * scale),
            output_height=int(total_height * scale)
        )
        print(f"Combined SVG and PNG saved at: {output_path.replace('.svg', '.png')}")

    except Exception as e:
        print(f"Failed to combine SVGs in {folder_path}: {e}")

for folder_name in os.listdir(output_base):
    folder_path = os.path.join(output_base, folder_name)
    if os.path.isdir(folder_path):
        combine_svgs_vertically(
            folder_path=folder_path,
            label_positions=[(130, 30), (490, 30), (850, 30), (130, 390), (490, 390), (850, 390)]
        )
