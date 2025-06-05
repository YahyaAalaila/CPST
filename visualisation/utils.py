import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import imageio
import os
from scipy.stats import gaussian_kde
import pickle
import matplotlib.patches as mpatches

def create_kde_rgb_gif(events, domain, T, bins=100, bandwidth=0.05, filename='kde_rgb.gif'):
    """
    Create a single GIF that color-codes the densities of mark=1 (red) and mark=2 (green)
    via kernel density estimation (KDE).
    
    events: list of dicts with keys: 't','x','y','type'
    domain: [x_min, x_max, y_min, y_max]
    T: total time horizon
    bins: resolution for the grid
    bandwidth: smoothing parameter for KDE
    filename: output GIF filename
    
    Approach:
      - For each time frame, gather mark=1 events -> KDE -> Z1
      - gather mark=2 events -> KDE -> Z2
      - Normalize each to [0,1]
      - Combine into an RGB image => (Z1, Z2, 0)
      - Save frames -> create GIF
    """
    x_min, x_max, y_min, y_max = domain
    
    # We'll create 50 frames from t=0 to t=T
    n_frames = 50
    times = np.linspace(0, T, n_frames)
    filenames = []
    
    # Prepare a grid for KDE evaluation
    x_grid = np.linspace(x_min, x_max, bins)
    y_grid = np.linspace(y_min, y_max, bins)
    xx, yy = np.meshgrid(x_grid, y_grid)
    grid_coords = np.vstack([xx.ravel(), yy.ravel()])  # shape (2, bins*bins)
    
    # Helper function for building the KDE array for a subset of events
    def kde_for_events(ev):
        if len(ev) < 2:
            return np.zeros((bins, bins))
        coords = np.array([[e['x'], e['y']] for e in ev]).T  # shape (2, n_events)
        cov = np.cov(coords)
        if np.linalg.matrix_rank(cov) < 2:
        # Option A: skip
            return np.zeros((bins,bins))
        kde = gaussian_kde(coords, bw_method=bandwidth)
        Z = kde(grid_coords).reshape(bins, bins)
        return Z
    
    for t in times:
        # Separate mark=1 and mark=2 events up to time t
        ev1 = [e for e in events if e['t'] <= t and e['type'] == 1]
        ev2 = [e for e in events if e['t'] <= t and e['type'] == 2]
        
        Z1 = kde_for_events(ev1)
        Z2 = kde_for_events(ev2)
        
        # Normalize each Z to [0,1] (you can also do percentile-based scaling if desired)
        z1_max = Z1.max() if Z1.size > 0 else 1.0
        z2_max = Z2.max() if Z2.size > 0 else 1.0
        if z1_max < 1e-12:  # avoid division by zero
            z1_max = 1.0
        if z2_max < 1e-12:
            z2_max = 1.0
        
        Z1_norm = Z1 / z1_max
        Z2_norm = Z2 / z2_max
        
        # Build an RGB image of shape (bins, bins, 3)
        rgb = np.zeros((bins, bins, 3))
        rgb[..., 0] = Z1_norm  # red channel
        rgb[..., 1] = Z2_norm  # green channel
        # Blue channel is 0 => no data there
        
        # Plot the RGB image
        plt.figure(figsize=(6,6))
        plt.imshow(rgb, origin='lower', extent=[x_min, x_max, y_min, y_max], aspect='auto')
        plt.title(f"Time = {t:.2f} (Mark1=Red, Mark2=Green)")
        
        temp_filename = f"_kde_rgb_frame_{t:.2f}.png"
        plt.savefig(temp_filename)
        plt.close()
        filenames.append(temp_filename)
    
    # Create GIF using imageio
    with imageio.get_writer(filename, mode='I', duration=0.2) as writer:
        for fn in filenames:
            image = imageio.imread(fn)
            writer.append_data(image)
    for fn in filenames:
        os.remove(fn)
    print(f"RGB KDE GIF saved as {filename}")


# -------------------------------
# Animation Function (Using Provided create_animation_gif)
# -------------------------------

def create_animation_gif(data, t_vals, fps=10, filename="animation.gif"):
    """
    Create and save an animation (as .gif) of the given 2D+time data.
    
    Parameters
    ----------
    data : np.ndarray, shape (Nt, Nx, Ny)
        The data to animate.
    t_vals : np.ndarray, shape (Nt,)
        Time values corresponding to data's first dimension.
    fps : int
        Frames per second.
    filename : str
        Output GIF filename.
    """
    num_frames = data.shape[0]
    fig, ax = plt.subplots(figsize=(5, 4))
    vmin, vmax = data.min(), data.max()
    im = ax.imshow(data[0], origin="lower", cmap="viridis", vmin=vmin, vmax=vmax, animated=True)
    ax.set_title(f"Time = {t_vals[0]:.2f}")
    
    def update(frame):
        im.set_data(data[frame])
        ax.set_title(f"Time = {t_vals[frame]:.2f}")
        return [im]
    
    ani = animation.FuncAnimation(fig, update, frames=range(num_frames), interval=1000/fps, blit=True)
    Writer = animation.PillowWriter
    writer = Writer(fps=fps)
    ani.save(filename, writer=writer)
    plt.close(fig)
    print(f"Animation GIF saved as {filename}")

def generate_heatmap_data(events, domain, T, Nt=50, bins=100):
    """
    Generate a 3D array (Nt, bins, bins) of 2D histograms for the events.
    
    Parameters:
      events : list of dicts with keys 't','x','y'
      domain : [x_min, x_max, y_min, y_max]
      T : total time horizon
      Nt : number of time frames
      bins : number of bins in each spatial dimension.
      
    Returns:
      data : np.ndarray of shape (Nt, bins, bins)
      t_vals : np.ndarray of time values.
    """
    x_min, x_max, y_min, y_max = domain
    t_vals = np.linspace(0, T, Nt)
    data = np.zeros((Nt, bins, bins))
    x_edges = np.linspace(x_min, x_max, bins+1)
    y_edges = np.linspace(y_min, y_max, bins+1)
    
    for i, t in enumerate(t_vals):
        ev = [e for e in events if e['t'] <= t]
        if ev:
            xs = np.array([e['x'] for e in ev])
            ys = np.array([e['y'] for e in ev])
            H, _, _ = np.histogram2d(xs, ys, bins=[x_edges, y_edges])
            data[i] = H
        else:
            data[i] = np.zeros((bins, bins))
    return data, t_vals
def create_gif(events, domain, T, filename='hawkes.gif'):
    import matplotlib.pyplot as plt
    import imageio
    
    x_min, x_max, y_min, y_max = domain
    # We'll generate frames at a sequence of time instants.
    times = np.linspace(0, T, 100)
    filenames = []
    for t in times:
        plt.figure(figsize=(6,6))
        # Plot events up to time t
        ev = [e for e in events if e['t'] <= t]
        if len(ev) > 0:
            xs = np.array([e['x'] for e in ev])
            ys = np.array([e['y'] for e in ev])
            types = np.array([e['type'] for e in ev])
            # Use different colors for different types:
            plt.scatter(xs[types==1], ys[types==1], color='blue', label='Type 1', alpha=0.6)
            plt.scatter(xs[types==2], ys[types==2], color='red', label='Type 2', alpha=0.6)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.title(f'Time = {t:.1f}')
        plt.legend()
        temp_filename = f'_frame_{t:.2f}.png'
        plt.savefig(temp_filename)
        plt.close()
        filenames.append(temp_filename)
    
    # Create GIF using imageio.
    with imageio.get_writer(filename, mode='I', duration=0.1) as writer:
        for temp_filename in filenames:
            image = imageio.imread(temp_filename)
            writer.append_data(image)
    
    # Remove temporary files
    for temp_filename in filenames:
        os.remove(temp_filename)
    print(f'GIF saved as {filename}')

def create_kde_heatmap_gif(events, domain, T, bins=100, bandwidth=0.05, filename='hawkes_kde_heatmap.gif'):
    """
    Create a GIF of smooth heatmaps (KDE) showing how the density of events evolves over time.
    
    Args:
      events: list of dicts with keys 't','x','y'
      domain: [x_min, x_max, y_min, y_max]
      T: total time horizon
      bins: resolution for evaluating KDE on a grid (bins x bins)
      bandwidth: smoothing parameter for the KDE
      filename: name of the output GIF
    """
    x_min, x_max, y_min, y_max = domain
    
    # We'll create 50 frames from t=0 to t=T
    n_frames = 50
    times = np.linspace(0, T, n_frames)
    filenames = []
    
    # Prepare a grid for KDE evaluation
    x_grid = np.linspace(x_min, x_max, bins)
    y_grid = np.linspace(y_min, y_max, bins)
    xx, yy = np.meshgrid(x_grid, y_grid)
    grid_coords = np.vstack([xx.ravel(), yy.ravel()])
    
    for t in times:
        # Gather events up to time t
        ev = [e for e in events if e['t'] <= t]
        
        # Evaluate the KDE if we have events
        if len(ev) > 0:
            coords = np.array([[e['x'], e['y']] for e in ev]).T  # shape (2, n_events)
            kde = gaussian_kde(coords, bw_method=bandwidth)
            Z = kde(grid_coords).reshape(bins, bins)
        else:
            Z = np.zeros((bins,bins))
        
        # Plot the KDE as an image
        plt.figure(figsize=(6,6))
        plt.imshow(Z.T, origin='lower', extent=[x_min, x_max, y_min, y_max],
                   aspect='auto', cmap='viridis')
        plt.colorbar(label='KDE Density')
        plt.title(f'Time = {t:.2f}')
        
        temp_filename = f'_kde_frame_{t:.2f}.png'
        plt.savefig(temp_filename)
        plt.close()
        filenames.append(temp_filename)
    
    # Create GIF using imageio
    import imageio
    with imageio.get_writer(filename, mode='I', duration=0.2) as writer:
        for fn in filenames:
            image = imageio.imread(fn)
            writer.append_data(image)
    
    # Clean up temporary files
    for fn in filenames:
        os.remove(fn)
    print(f"KDE heatmap GIF saved as {filename}")

        # Complexity measure (for the Hawkes kernel portion)
def compute_Phi(beta, sigma):
    return beta / (2 * np.pi * sigma**2)

def complexity_bound(B, A, Phi_k, n):
    op_norm_A = np.linalg.norm(A, ord=2)
    return B * op_norm_A * np.sqrt(Phi_k / n)
def create_scatter_fade_gif(events, domain, T, tau=20.0, fps=10, filename="scatter_fade.gif"):
    """
    Create a scatter plot GIF where events fade out after a certain time.
    Each event's transparency is given by alpha = exp(-(t_current - t_event)/tau).
    
    Parameters:
      events: list of dicts with keys 't', 'x', 'y', 'type'
      domain: [x_min, x_max, y_min, y_max]
      T: total time horizon
      tau: decay time constant for fading (e.g., reflecting GDP decay time)
      fps: frames per second
      filename: output GIF filename
    """
    x_min, x_max, y_min, y_max = domain
    n_frames = 100
    times = np.linspace(0, T, n_frames)
    filenames = []
        
    red_patch = mpatches.Patch(color='red', label='Mark 1')
    green_patch = mpatches.Patch(color='green', label='Mark 2')
    blue_patch = mpatches.Patch(color='blue', label='Mark 3')
    custom_handles = [red_patch, green_patch, blue_patch]
    for t in times:
        plt.figure(figsize=(6,6))
        # For each event, if t_event is older than t, compute age = t - t_event and alpha = exp(-age/tau)
        # Only plot events if alpha is above a threshold (e.g., 0.05)
        xs1, ys1, alphas1 = [], [], []
        xs2, ys2, alphas2 = [], [], []
        xs3, ys3, alphas3 = [], [], []
        for e in events:
            if e['t'] <= t:
                age = t - e['t']
                alpha = np.exp(-age/tau)
                if alpha < 0.05:
                    continue  # skip very faded events
                if e['type'] == 1:
                    xs1.append(e['x'])
                    ys1.append(e['y'])
                    alphas1.append(alpha)
                elif e['type'] == 2:
                    xs2.append(e['x'])
                    ys2.append(e['y'])
                    alphas2.append(alpha)
                elif e['type'] == 3:
                    xs3.append(e['x'])
                    ys3.append(e['y'])
                    alphas3.append(alpha)
        # Plot events for each mark with corresponding colors and alpha
        if xs1:
            plt.scatter(xs1, ys1, color="red", alpha=0.8, s=20, label="Mark 1", edgecolors='none')
            # To incorporate fading, we plot in a loop with different alphas:
            for x, y, a in zip(xs1, ys1, alphas1):
                plt.scatter(x, y, color="red", alpha=a, s=20, edgecolors='none')
        if xs2:
            for x, y, a in zip(xs2, ys2, alphas2):
                plt.scatter(x, y, color="green", alpha=a, s=20, edgecolors='none')
        if xs3:
            for x, y, a in zip(xs3, ys3, alphas3):
                plt.scatter(x, y, color="blue", alpha=a, s=20, edgecolors='none')
        
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.title(f"Time = {t:.2f}")
        plt.legend(handles=custom_handles, loc="upper right")
        temp_filename = f"_scatter_frame_{t:.2f}.png"
        plt.savefig(temp_filename)
        plt.close()
        filenames.append(temp_filename)
    
    # Create GIF using imageio
    with imageio.get_writer(filename, mode="I", duration=1.0/fps) as writer:
        for fn in filenames:
            image = imageio.imread(fn)
            writer.append_data(image)
    for fn in filenames:
        os.remove(fn)
    print(f"Scatter fade GIF saved as {filename}")



def create_scatter_fade_multi_gif(
    events_list,   # List of event-lists, each for a different 'complexity' or dataset
    domain,        # [x_min, x_max, y_min, y_max]
    T,             # Total time horizon
    tau=20.0,      # Decay constant for fading
    fps=10,        # Frames per second
    filename="scatter_fade_multi.gif"
):
    """
    Create a single GIF with multiple subplots side-by-side, each showing a scatter
    plot of a different dataset's events (with 3 possible marks).
    Events fade out after time 'tau' via alpha = exp(-(t_current - t_event)/tau).

    Parameters
    ----------
    events_list : list of lists
        Each element is a list of event dicts for one dataset.
        e.g. events_list[i][j] => an event with keys {'t','x','y','type'}
    domain : list of float
        [x_min, x_max, y_min, y_max]
    T : float
        total time horizon
    tau : float
        fade time constant
    fps : float
        frames per second in final GIF
    filename : str
        name of the output GIF
    """
    # Each dataset in events_list can have a different number of events.
    # We'll produce a single figure with len(events_list) subplots per frame.
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    num_datasets = len(events_list)
    x_min, x_max, y_min, y_max = domain
    
    # We'll produce 100 frames from t=0 to t=T
    n_frames = 100
    times = np.linspace(0, T, n_frames)
    filenames = []

    # Prepare custom legend handles for each subplot (3 marks).
    red_patch   = mpatches.Patch(color='red',   label='Mark 1')
    green_patch = mpatches.Patch(color='green', label='Mark 2')
    blue_patch  = mpatches.Patch(color='blue',  label='Mark 3')
    custom_handles = [red_patch, green_patch, blue_patch]

    for frame_i, t in enumerate(times):
        # Create a figure with subplots: 1 row, num_datasets columns
        fig, axes = plt.subplots(1, num_datasets, figsize=(6*num_datasets, 5), sharex=True, sharey=True)
        if num_datasets == 1:
            # If there's only one dataset, axes is just a single Axes object
            axes = [axes]
        
        for ds_i, ax in enumerate(axes):
            # We'll gather events up to time t for the ds_i-th dataset
            evs = events_list[ds_i]
            # We'll store separate lists for each mark
            xs1, ys1, alphas1 = [], [], []
            xs2, ys2, alphas2 = [], [], []
            xs3, ys3, alphas3 = [], [], []

            for e in evs:
                if e['t'] <= t:
                    age = t - e['t']
                    alpha = np.exp(-age/tau)
                    if alpha < 0.05:
                        continue  # skip very faded events
                    if e['type'] == 1:
                        xs1.append(e['x'])
                        ys1.append(e['y'])
                        alphas1.append(alpha)
                    elif e['type'] == 2:
                        xs2.append(e['x'])
                        ys2.append(e['y'])
                        alphas2.append(alpha)
                    elif e['type'] == 3:
                        xs3.append(e['x'])
                        ys3.append(e['y'])
                        alphas3.append(alpha)

            # Now we plot them. For fading each event individually, we do a loop:
            # (Alternatively, you can do one scatter call per mark with an average alpha.)
            for x, y, a in zip(xs1, ys1, alphas1):
                ax.scatter(x, y, color="red", alpha=a, s=20, edgecolors='none')
            for x, y, a in zip(xs2, ys2, alphas2):
                ax.scatter(x, y, color="green", alpha=a, s=20, edgecolors='none')
            for x, y, a in zip(xs3, ys3, alphas3):
                ax.scatter(x, y, color="blue", alpha=a, s=20, edgecolors='none')

            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_title(f"data_{ds_i}", fontsize=10)
            ax.legend(handles=custom_handles, loc="upper right")

        fig.suptitle(f"Time = {t:.2f}", fontsize=12)
        temp_filename = f"_scatter_multi_{frame_i:03d}.png"
        plt.savefig(temp_filename, dpi=100)
        plt.close(fig)
        filenames.append(temp_filename)

    # Combine frames into a GIF
    with imageio.get_writer(filename, mode="I", duration=1.0/fps) as writer:
        for fn in filenames:
            image = imageio.imread(fn)
            writer.append_data(image)

    # Clean up temporary PNGs
    for fn in filenames:
        os.remove(fn)
    
    print(f"Scatter fade multi-GIF saved as {filename}")
