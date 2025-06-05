# CPST: Complex Patterns in Spatio-Temporal Data

This repository contains a small collection of modules to generate synthetic spatio-temporal point patterns using Hawkes processes. Background and triggering components can be combined to produce data with different levels of complexity. Utilities are provided to visualise the generated events.

## Repository layout

```
complexity/          # placeholder for future complexity measures
config/              # example YAML configuration
examples/            # scripts demonstrating how to run simulations
intensities/         # background and kernel intensity definitions
simulators/          # data generation classes (e.g. MultiMarkHawkesDGP)
utils/               # generic utilities (thinning algorithm)
visualisation/       # helper functions to create GIFs and plots
```

Many parts are still experimental, but the structure is modular so different backgrounds or kernels can be swapped via configuration.

## Quick start

Install the dependencies (for example: `numpy`, `pandas`, `omegaconf`, `joblib`, `tqdm`, `matplotlib`, `imageio`). Then run one of the example scripts:

```bash
python examples/example1.py                       # single run
python examples/example2.py -c examples/config.yaml --jobs 4  # parameter sweep
```

Parameters and intensity options are stored in YAML files under `config/` or `examples/`. The Hawkes simulator constructs components via `intensities.registry`.

Generated events can be stored as pickles or parquet files and visualised with the helpers in `visualisation/utils.py`.

For a list of potential improvements and project notes see [`docs/REPO_REVIEW.md`](docs/REPO_REVIEW.md).

## License

This repository is licensed under the Apache 2.0 license.
