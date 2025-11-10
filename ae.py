'''

q: daniel sinausia

'''


import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import random
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import uniform_filter1d
import matplotlib as mpl

mpl.use('SVG')
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Open Sans', 'Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']

def set_seeds(seed=42):
    """Set seeds for reproducible results across all random number generators"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Make CuDNN deterministic (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=3):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z
    
    def encode(self, x):
        """Encode input to latent space"""
        with torch.no_grad():
            return self.encoder(x)
    
    def decode(self, z):
        """Decode from latent space to reconstruction"""
        with torch.no_grad():
            return self.decoder(z)
          
def plot_latent_features_over_time(Z_df, save_path=None, save_svg=True, title_prefix=""):
    """
    Plot all latent features over time in a single plot
    
    Args:
        Z_df: DataFrame with 'Time (s)' and latent dimensions (Z1, Z2, Z3, etc.)
        save_path: Optional path to save the figure (without extension)
        save_svg: Whether to save as SVG in addition to PNG
        title_prefix: Prefix for the plot title
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    time_axis = Z_df['Time (s)']

    latent_cols = [col for col in Z_df.columns if col.startswith('Z')]
    colors = plt.cm.Blues(np.linspace(0.3, 1, len(latent_cols)))
    
    for i, col in enumerate(latent_cols):
        color = colors[i]
        ax.plot(time_axis, Z_df[col], color=color, alpha=0.8, 
                linewidth=2, label=f'{col}')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Latent Value')
    ax.set_title(f'{title_prefix}Latent Features Evolution Over Time')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_xlim(time_axis.min(), time_axis.max())
    
    plt.tight_layout()
    
    if save_path:
        save_path_str = str(save_path)
        png_path = f"{save_path_str}.png" if not save_path_str.endswith('.png') else save_path_str
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        print(f"Latent features plot saved (PNG): {png_path}")
        if save_svg:
            svg_path = png_path.replace('.png', '.svg')
            plt.savefig(svg_path, format='svg', bbox_inches='tight')
            print(f"Latent features plot saved (SVG): {svg_path}")
    
    return fig

def plot_training_loss(epoch_losses, save_path=None, save_svg=True, title_prefix=""):
    """
    Plot the evolution of training loss with professional styling
    
    Args:
        epoch_losses: List of average losses per epoch
        save_path: Optional path to save the figure (without extension)
        save_svg: Whether to save as SVG in addition to PNG
        title_prefix: Prefix for the plot title
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    epochs = range(1, len(epoch_losses) + 1)
    mean_loss = np.mean(epoch_losses)
    ax.plot(epochs, epoch_losses, color='#1f77b4', alpha=0.8, linewidth=2, label='Training Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(f'Training History: Mean {mean_loss:.0f}')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_xlim(min(epochs), max(epochs))
    
    plt.tight_layout()
    
    if save_path:
        save_path_str = str(save_path)
        png_path = f"{save_path_str}.png" if not save_path_str.endswith('.png') else save_path_str
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        print(f"Loss plot saved (PNG): {png_path}")
        if save_svg:
            svg_path = png_path.replace('.png', '.svg')
            plt.savefig(svg_path, format='svg', bbox_inches='tight')
            print(f"Loss plot saved (SVG): {svg_path}")
    
    return fig

def traverse_latent_dimension(model, base_spectrum, dim_idx, traversal_range=(-2, 2), n_steps=5, device='cpu'):
    """
    Traverse a single latent dimension while keeping others fixed
    
    Args:
        model: Trained autoencoder model
        base_spectrum: Reference spectrum to start traversal from (1D array or tensor)
        dim_idx: Index of latent dimension to traverse (0, 1, 2, etc.)
        traversal_range: Tuple of (min_val, max_val) for traversal
        n_steps: Number of steps in the traversal
        device: torch device
    
    Returns:
        traversal_values: Array of latent values used for traversal
        generated_spectra: Array of generated spectra (n_steps x spectrum_length)
        latent_codes: Array of latent codes used (n_steps x latent_dim)
    """
    model.eval()
    if isinstance(base_spectrum, np.ndarray):
        base_spectrum = torch.tensor(base_spectrum, dtype=torch.float32)
    base_spectrum = base_spectrum.to(device)
    if base_spectrum.dim() == 1:
        base_spectrum = base_spectrum.unsqueeze(0)
    base_latent = model.encode(base_spectrum).squeeze()
    traversal_values = np.linspace(traversal_range[0], traversal_range[1], n_steps)
    
    generated_spectra = []
    latent_codes = []
    
    with torch.no_grad():
        for val in traversal_values:
            modified_latent = base_latent.clone()
            modified_latent[dim_idx] = val
            modified_latent = modified_latent.unsqueeze(0)  # Add batch dim
            generated_spectrum = model.decode(modified_latent).squeeze()
            
            generated_spectra.append(generated_spectrum.cpu().numpy())
            latent_codes.append(modified_latent.squeeze().cpu().numpy())
    
    return traversal_values, np.array(generated_spectra), np.array(latent_codes)

def plot_latent_traversal(wavenumbers, traversal_values, generated_spectra, dim_idx, 
                         title_prefix="", save_path=None, save_svg=True, figsize=(12, 8)):
    """
    Plot the results of latent dimension traversal
    
    Args:
        wavenumbers: Wavenumber axis for spectra
        traversal_values: Array of latent values used for traversal
        generated_spectra: Array of generated spectra
        dim_idx: Index of the traversed dimension
        title_prefix: Prefix for the plot title
        save_path: Optional path to save the figure (without extension)
        save_svg: Whether to save as SVG in addition to PNG
        figsize: Figure size tuple
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    x_min, x_max = wavenumbers.min(), wavenumbers.max()
    colors = plt.cm.Blues(np.linspace(0.3, 1, len(traversal_values)))
    
    for i, (spectrum, val, color) in enumerate(zip(generated_spectra, traversal_values, colors)):
        show_label = len(traversal_values) <= 10 or i % 2 == 0
        label = f'Z{dim_idx+1}={val:.1f}' if show_label else ''
        
        ax1.plot(wavenumbers, spectrum, color=color, alpha=0.8, 
                linewidth=2, label=label)
    
    ax1.set_xlabel('Wavenumber (cm⁻¹)')
    ax1.set_ylabel('Intensity')
    ax1.set_title(f'{title_prefix}Latent Dimension Z{dim_idx+1} Traversal')
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.set_xlim(x_max, x_min)
    base_spectrum = generated_spectra[0]
    for i, (spectrum, val, color) in enumerate(zip(generated_spectra[1:], 
                                                  traversal_values[1:], 
                                                  colors[1:])):
        diff = spectrum - base_spectrum
        ax2.plot(wavenumbers, diff, color=color, alpha=0.8, linewidth=2,
                label=f'Z{dim_idx+1}={val:.1f}')
    
    ax2.set_xlabel('Wavenumber (cm⁻¹)')
    ax2.set_ylabel('Intensity Difference')
    ax2.set_title(f'Difference Spectra (relative to Z{dim_idx+1}={traversal_values[0]:.1f})')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.set_xlim(x_max, x_min)
    
    plt.tight_layout()
    
    if save_path:
        save_path_str = str(save_path)
        png_path = f"{save_path_str}.png" if not save_path_str.endswith('.png') else save_path_str
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        print(f"Traversal plot saved (PNG): {png_path}")

        if save_svg:
            svg_path = png_path.replace('.png', '.svg')
            plt.savefig(svg_path, format='svg', bbox_inches='tight')
            print(f"Traversal plot saved (SVG): {svg_path}")
    
    return fig


def perform_all_traversals(model, base_spectrum, wavenumbers, latent_dim, 
                          save_dir=None, save_svg=True, title_prefix="", device='cpu'):
    """
    Perform traversal for all latent dimensions and create plots
    
    Args:
        model: Trained autoencoder
        base_spectrum: Reference spectrum for traversal
        wavenumbers: Wavenumber axis
        latent_dim: Number of latent dimensions
        save_dir: Directory to save plots and data
        save_svg: Whether to save plots as SVG in addition to PNG
        title_prefix: Prefix for plot titles
        device: torch device
    
    Returns:
        results: Dictionary with traversal results for each dimension
    """
    results = {}
    
    for dim_idx in range(latent_dim):
        print(f"Traversing latent dimension Z{dim_idx+1}...")
        
        traversal_vals, generated_specs, latent_codes = traverse_latent_dimension(
            model, base_spectrum, dim_idx, device=device
        )
        results[f'Z{dim_idx+1}'] = {
            'traversal_values': traversal_vals,
            'generated_spectra': generated_specs,
            'latent_codes': latent_codes
        }

        if save_dir:
            save_path = Path(save_dir) / f"traversal_Z{dim_idx+1}"
            save_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            save_path = None
            
        fig = plot_latent_traversal(
            wavenumbers, traversal_vals, generated_specs, dim_idx,
            title_prefix=title_prefix, save_path=save_path, save_svg=save_svg
        )
        if save_dir:
            data_path = Path(save_dir) / f"traversal_Z{dim_idx+1}_data.csv"
            df_traversal = pd.DataFrame({
                'latent_value': traversal_vals,
                **{f'wavenumber_{i}': generated_specs[:, i] 
                   for i in range(generated_specs.shape[1])}
            })
            df_traversal.to_csv(data_path, index=False)
            print(f"Traversal data saved: {data_path}")
        
        plt.show()
    
    return results

def process_file(file_path, device, seed=42, perform_traversal=True, plot_loss=True, save_svg=True):
    set_seeds(seed)
    
    df = pd.read_csv(file_path, index_col=0)
    spectra = df.to_numpy().T.astype(np.float32).copy(order="C")
    wavenumbers = df.index.values.astype(float)

    print("Spectra shape (time, wavenumber):", spectra.shape)
    print("Estimated memory usage: {:.2f} MB".format(spectra.nbytes / 1e6))

    time_axis = [float(c) for c in df.columns]
    spectra_normalized = (spectra - spectra.mean(axis=0)) / (spectra.std(axis=0) + 1e-8)
    spectra_tensor = torch.tensor(spectra_normalized, dtype=torch.float32).to(device)

    dataset = DataLoader(TensorDataset(spectra_tensor), batch_size=4, shuffle=True)
    latent_dim = 3
    model = Autoencoder(input_dim=spectra.shape[1], latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    model.train()
    epoch_losses = []
    
    for epoch in range(30):
        epoch_loss = 0
        batch_count = 0
        for batch in dataset:
            x = batch[0]
            optimizer.zero_grad()
            x_recon, _ = model(x)
            loss = loss_fn(x_recon, x)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            batch_count += 1
        avg_epoch_loss = epoch_loss / batch_count
        epoch_losses.append(avg_epoch_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/30, Loss: {avg_epoch_loss:.6f}")
    if plot_loss:
        print("\n Plotting training loss evolution...")
        file_dir = Path(file_path).parent
        loss_save_path = file_dir / "training_loss"
        
        fig_loss = plot_training_loss(
            epoch_losses, 
            save_path=loss_save_path,
            save_svg=save_svg,
            title_prefix=f"{file_dir.name} - "
        )
        
        loss_data_path = file_dir / "training_loss_data.csv"
        loss_df = pd.DataFrame({
            'epoch': range(1, len(epoch_losses) + 1),
            'average_loss': epoch_losses
        })
        loss_df.to_csv(loss_data_path, index=False)
        print(f"Loss data saved: {loss_data_path}")
        
        plt.show()
    model.eval()
    Z_all = []
    with torch.no_grad():
        for x in DataLoader(TensorDataset(spectra_tensor), batch_size=1):
            _, z = model(x[0])
            Z_all.append(z.cpu())

    Z = torch.cat(Z_all, dim=0).numpy()
    Z_df = pd.DataFrame({
        "Time (s)": time_axis,
        **{f"Z{i+1}": Z[:, i] for i in range(Z.shape[1])}
    })
    if plot_loss:
        print("\n Plotting latent features over time...")

        file_dir = Path(file_path).parent
        latent_save_path = file_dir / "latent_features_over_time"

        fig_latent = plot_latent_features_over_time(
            Z_df, 
            save_path=latent_save_path,
            save_svg=save_svg,
            title_prefix=f"{file_dir.name} - "
        )

        plt.show()

    if perform_traversal:
        print("\n Performing latent dimension traversal...")

        middle_idx = len(spectra_normalized) // 2
        base_spectrum = spectra_normalized[middle_idx]

        file_dir = Path(file_path).parent
        traversal_dir = file_dir / "traversal_analysis"

        traversal_results = perform_all_traversals(
            model=model,
            base_spectrum=base_spectrum,
            wavenumbers=wavenumbers,
            latent_dim=latent_dim,
            save_dir=traversal_dir,
            save_svg=save_svg,
            title_prefix=f"{file_dir.name} - ",
            device=device
        )
        
        print(f"Traversal analysis completed and saved to: {traversal_dir}")

    return Z_df

def process_all_subfolders(base_dir, global_seed=42, perform_traversal=True, plot_loss=True, save_svg=True):
    set_seeds(global_seed)
    
    device = torch.device("cpu")
    total = 0
    for root, _, files in os.walk(base_dir):
        if "spectral_matrix.csv" in files:
            file_path = os.path.join(root, "spectral_matrix.csv")
            print(f"\n🔍 Found and processing: {file_path}")
            try:
                file_seed = global_seed + total
                Z_df = process_file(file_path, device, seed=file_seed, 
                                  perform_traversal=perform_traversal, plot_loss=plot_loss,
                                  save_svg=save_svg)
                save_path = os.path.join(root, "AE_latent_Z.csv")
                Z_df.to_csv(save_path, index=False)
                print(f"Saved: {save_path}")
                total += 1
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    print(f"\n Total files processed: {total}")

if __name__ == "__main__":
    base_directory = '...'

    MAIN_SEED = 42
    process_all_subfolders(base_directory, global_seed=MAIN_SEED, perform_traversal=True, plot_loss=True, save_svg=True)
