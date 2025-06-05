from datetime import datetime

import os
import pickle
from omegaconf import OmegaConf
import numpy as np
from simulators.Hawkes import MultiMarkHawkesDGP
from visualisation.utils import create_scatter_fade_multi_gif, compute_Phi, complexity_bound

def load_config(config_file: str):
    # Load the YAML configuration file using OmegaConf
    config = OmegaConf.load(config_file)
    return config
def save_run_output(events, X, viz_function, background_config, kernel_config, domain, T, tp):
    """
    Save both the simulation data and the visualization output in a run-specific folder.
    The folder name encodes the background type, kernel type, and kernel mode.
    
    Parameters:
        events: list of event dicts.
        X: array of spatio-temporal points.
        viz_function: function to generate visualization (e.g. a GIF).
        background_config: dict specifying the background configuration.
        kernel_config: dict specifying the kernel configuration.
        domain: list [x_min, x_max, y_min, y_max].
        T: total simulation time.
        run_config: dict containing additional run parameters (like run_id).
    """
    # Extract key parameters from configurations.
    btype = background_config.get("type", "constant")
    kmode = kernel_config.get("mode", "separate")
    
    # Create a unique run id; incorporate background and kernel information.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #run_id = run_config.get("run_id", f"{timestamp}")
    
    # Build a folder name that reflects the configuration.
    folder_name = f"{btype}_{kmode}_{timestamp}"
    
    # Create directory structure: results/<folder_name>/data and /visualizations
    root_folder = os.path.join("results", folder_name)

    
    # Save simulation data as a pickle file.
    if tp == 'data':
        data_folder = os.path.join(root_folder, "data")
        os.makedirs(data_folder, exist_ok=True)
        data_filepath = os.path.join(data_folder, "simulation_data.pkl")
        with open(data_filepath, 'wb') as f:
            pickle.dump((events, X), f)
        print(f"Simulation data saved at: {data_filepath}")
    if tp == 'viz':
        # Save visualization output.
        viz_folder = os.path.join(root_folder, "visualizations")
        os.makedirs(viz_folder, exist_ok=True)
        viz_filepath = os.path.join(viz_folder, "simulation_visualization.gif")
        viz_function(events, domain, T, filename=viz_filepath)
        print(f"Visualization saved at: {viz_filepath}")

def main():
    # Load configuration from YAML file
    config = load_config("config/config.yaml")
    
    # Access configuration using attribute notation
    sim_config = config.simulation
    norm_config = config.normalization
    kernel_config = config.kernel
    
    # Use defaults if a parameter is missing
    T = sim_config.T if 'T' in sim_config else 10.0
    domain = sim_config.domain if 'domain' in sim_config else [0, 10, 0, 10]
    A = np.array(kernel_config.A if 'A' in kernel_config else 1)
    Lambda = sim_config.Lambda if 'Lambda' in sim_config else 10.0
    print("A:", A)
    
    mean = np.array(norm_config.mean if 'mean' in norm_config else [5.0, 5.0])
    cov = np.array(norm_config.cov if 'cov' in norm_config else [
        [0.5, 0.8, 0.2],
        [0.0, 0.1, 0.3],
        [0.0, 0.0, 0.05]
    ])

    beta_param = 1.0
    sigma_param = 0.05
    # Define our 3x3 matrices for the branching matrix.
    A1 = np.array([[0.05, 0.0, 0.0],
                [0.0, 0.05, 0.0],
                [0.0, 0.0, 0.1]])

    A2 = np.array([[0.2, 0.5, 0.0],
                [0., 0.1, 0.05],
                [0.0, 0., 0.3]])

    A3 = np.array([[0.3, 0.4, 0.2],
                [0., 0.3, 0.25],
                [0., 0., 0.3]])

    A4 = np.array([[0.5, 0.5, 0.35],
                [0, 0.5, 0.4],
                [0., 0., 0.4]])

    # You can store these in a list for later use:
    A_matrices = [A1, A2, A3, A4]
    gen_data = True
    # Choose one for simulation. You can also experiment by switching.
    ev_data = []
    for name, A in enumerate(A_matrices):  # or A_low
    # Simulate two-mark Hawkes process with a deterministic background
        if gen_data:
            # Create the simulator using configurations
            simulator = MultiMarkHawkesDGP(
                T=T,
                domain=domain,
                A=A,
                Lambda=Lambda,
                background_config=config.background if 'background' in config else {},
                kernel_config=kernel_config,
                mean=mean,
                cov=cov
            )
            # Generate events
            events = simulator.generate_data()
            X = np.array([[e['t'], e['x'], e['y']] for e in events])

            save_run_output(events, X, create_scatter_fade_multi_gif, config.background , kernel_config, domain, T, tp='data')
        else:
            filepath = os.path.join("results", rf"simulation_data_3ev_{name}.pkl")
            with open(filepath, 'rb') as f:
                events, X = pickle.load(f)

        ev_data.append(events)
        print(f"Simulated {len(events)} total events.")
      
        Phi_k = compute_Phi(beta_param, sigma_param)
        B = beta_param / (2 * np.pi * sigma_param**2)

        comp_bound = complexity_bound(B, A, Phi_k, len(events))
        print(f"Complexity bound: {comp_bound:.4f}")

    #create_scatter_fade_multi_gif(ev_data, domain, T, tau=20.0, fps=10, filename=os.path.join("viz_scatter_fade_multi", rf"hawks_scatterfade_thining_100v2_{name}.gif"))
    save_run_output(ev_data, X, create_scatter_fade_multi_gif, config.background , kernel_config, domain, T, tp='viz')

if __name__ == "__main__":
    main()
