import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch


COLOR_CODES = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
# This is matplotlib tab10 with gray moved to the last position and all 0.5 alpha
PROBE_PLOT_COLORS = np.array([
        [0.12156863, 0.46666667, 0.70588235, 0.5],
        [1.        , 0.49803922, 0.05490196, 0.5],
        [0.17254902, 0.62745098, 0.17254902, 0.5],
        [0.83921569, 0.15294118, 0.15686275, 0.5],
        [0.58039216, 0.40392157, 0.74117647, 0.5],
        [0.54901961, 0.3372549 , 0.29411765, 0.5],
        [0.89019608, 0.46666667, 0.76078431, 0.5],
        [0.7372549 , 0.74117647, 0.13333333, 0.5],
        [0.09019608, 0.74509804, 0.81176471, 0.5],
        [0.49803922, 0.49803922, 0.49803922, 0.25]
    ])


def plot_drift_amount(ops, results_dir):
    plt.style.use('dark_background')
    fig, ax = plt.subplots(1, 1, figsize=(8,8))
    dshift = ops['dshift']
    settings = ops['settings']

    fs = settings['fs']
    NT = settings['batch_size']
    t = np.arange(dshift.shape[0])*(NT/fs)
    for i in range(dshift.shape[1]):
        color = COLOR_CODES[i % len(COLOR_CODES)]
        ax.plot(t, dshift[:,i], c=color)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Depth shift (um)')
    fig.suptitle('Drift amount per probe section, across batches')
    fig.tight_layout()

    save_path = results_dir / 'drift_amount.png'
    fig.savefig(save_path, dpi=300)
    plt.style.use('default')
    plt.close(fig)


def plot_drift_scatter(st0, results_dir):
    fig, ax = plt.subplots(1, 1, figsize=(30,14))

    x = st0[:,0]  # spike time in seconds
    y = st0[:,1]  # depth of spike center in microns
    z = st0[:,2]  # spike amplitude (data)
    z[z < 10] = 10
    z[z > 100] = 100
    colors = np.empty((x.shape[0], 4), dtype=float)

    bin_idx = np.digitize(z, np.logspace(1, 2, 90))
    cm = matplotlib.colormaps['binary']
    for i in np.unique(bin_idx):
        # Take mean of all amplitude values within one bin, map to color
        subset = (bin_idx == i)
        a = z[subset].mean()
        colors[subset] = cm(((a-10)/90))
    
    # Scatter of spike depth over time, with color intensity proportional
    # to log of amplitude.
    ax.scatter(x, y, s=3, c=colors)
    ax.set_xlabel('Time (s)', fontsize=22)
    ax.set_ylabel('Depth (um)', fontsize=22)
    fig.suptitle('Spike amplitude across time and depth', fontsize=30)
    fig.tight_layout()

    save_path = results_dir / 'drift_scatter.png'
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_diagnostics(Wall0, clu0, ops, results_dir):
    plt.style.use('dark_background')
    fig, axes = plt.subplots(2, 2, figsize=(16,16))
    wPCA = ops['wPCA']
    settings = ops['settings']

    # Top left
    t = np.arange(wPCA.shape[1])/(settings['fs']/1000)
    for i in range(wPCA.shape[0]):
        color = COLOR_CODES[i % len(COLOR_CODES)]
        axes[0][0].plot(t, wPCA[i,:].cpu().numpy(), c=color)
    axes[0][0].set_xlabel('Time (s)')
    axes[0][0].set_title('Temporal Features')

    # Top right
    features = torch.linalg.norm(Wall0, dim=2).cpu().numpy()
    axes[0][1].imshow(features.T, aspect='auto', vmin=0, vmax=25, cmap='binary_r')
    axes[0][1].set_xlabel('Channel Number')
    axes[0][1].set_ylabel('Unit Number')
    axes[0][1].set_title('Spatial Features')

    # Comput spike counts and mean amplitudes
    n_units = int(clu0.max()) + 1
    spike_counts = np.zeros(n_units)
    for i in range(n_units):
        spike_counts[i] = (clu0[clu0 == i]).size
    mean_amp = torch.linalg.norm(Wall0, dim=(1,2)).cpu().numpy()

    # Bottom left
    axes[1][0].plot(mean_amp)
    axes[1][0].set_xlabel('Unit Number')
    axes[1][0].set_ylabel('Amplitude (a.u.)')
    axes[1][0].set_title('Unit Amplitudes')

    # Bottom right
    axes[1][1].scatter(np.log(1 + spike_counts), mean_amp, s=3)
    axes[1][1].set_xlabel('Log(1 + Spike Count)')
    axes[1][1].set_ylabel('Amplitude (a.u.)')
    axes[1][1].set_title('Amplitude vs Spike Count')

    fig.tight_layout()
    save_path = results_dir / 'diagnostics.png'
    fig.savefig(save_path, dpi=300)
    plt.style.use('default')
    plt.close(fig)


def plot_spike_positions(clu, is_refractory, results_dir):
    plt.style.use('dark_background')
    fig, ax = plt.subplots(1, 1, figsize=(30,14))

    # 10 colors in palette, last one is gray for non-frefractory
    clu = clu.copy()
    bad_units = np.unique(clu)[is_refractory == 0]
    bad_idx = np.in1d(clu, bad_units)
    clu = np.mod(clu, 9)
    clu[bad_idx] = 9
    colors = np.empty((clu.shape[0], 4), dtype=float)

    # Map modded cluster ids to colors
    for i in range(10):
        subset = (clu == i)
        rgba = PROBE_PLOT_COLORS[i]
        colors[subset] = rgba

    # Get x, y positions, add to scatterplot
    positions = np.load(results_dir / 'spike_positions.npy')
    xs, ys = positions[:,0], positions[:,1]
    ax.scatter(ys, xs, s=3, c=colors)
    ax.set_xlabel('Depth (um)', fontsize=22)
    ax.set_ylabel('Lateral (um)', fontsize=22)
    fig.suptitle('Spike position across probe, colored by cluster', fontsize=30)
    fig.tight_layout()

    save_path = results_dir / 'spike_positions.png'
    fig.savefig(save_path, dpi=300)
    plt.style.use('default')
    plt.close(fig)
