from .background.backgrounds import ConstantIntensity, DeterministicIntensity, PiecewiseIntensity
from .kernels.temporal.temporal_kernels import ExponentialTimeKernel
from .kernels.spatial.spatial_kernels import GaussianSpaceKernel
from .kernels.spatiotemporal.kernels import PolynomialJointKernel
from .kernels.parametric import ParametricKernel

def get_background(config: dict):
    """
    Create a background intensity instance based on a configuration dictionary.
    
    The config dict should include a "type" key with one of:
      - "constant"
      - "deterministic"
      - "piecewise"
    
    For example:
      background_config = {
          "type": "piecewise",
          "time_partitions": [0, 5, 10],
          "space_partitions": [
              {"x_min": 0, "x_max": 5, "y_min": 0, "y_max": 10},
              {"x_min": 5, "x_max": 10, "y_min": 0, "y_max": 10}
          ],
          "values_dict": {(0, 0): 2.0, (0, 1): 4.0, (1, 0): 1.0, (1, 1): 2.0}
      }
    """
    btype = config.get("type", "constant")
    if btype == "constant":
        c = config.get("c", 1.0)
        return ConstantIntensity(c)
    elif btype == "deterministic":
        a0 = config.get("a0", 0.0)
        a1 = config.get("a1", 0.0)
        omega = config.get("omega", 1.0)
        s0 = config.get("s0", [0, 0])
        sigma0 = config.get("sigma0", 1.0)
        return DeterministicIntensity(a0, a1, omega, s0, sigma0)
    elif btype == "piecewise":
        time_parts = config.get("time_partitions")
        space_parts = config.get("space_partitions")
        values_dict = config.get("values_dict")
        if time_parts is None or space_parts is None or values_dict is None:
            raise ValueError("For piecewise background, 'time_partitions', 'space_partitions', and 'values_dict' must be provided.")
        return PiecewiseIntensity(time_parts, space_parts, values_dict)
    else:
        raise ValueError(f"Unknown background type: {btype}")

def get_kernel(config: dict):
    """
    Create a kernel (triggering) instance based on a configuration dictionary.
    
    The config dict should include a "mode" key with either:
      - "separate": uses separate time and space kernels (product model)
      - "entangled": uses a joint (e.g., polynomial) kernel
    
    For example, for separate mode:
      kernel_config = {
          "mode": "separate",
          "time_kernel": {"type": "exponential", "beta": 1.0},
          "space_kernel": {"type": "gaussian", "sigma": 1.0},
          "A": [[0.5]]
      }
    For entangled mode:
      kernel_config = {
          "mode": "entangled",
          "joint_kernel": {
              "type": "polynomial",
              "coefficients": {(0,0,0): 1.0, (1,0,0): 0.5}
          }
      }
    """
    mode = config.get("mode", "separate")
    if mode == "separate":
        time_cfg = config.get("time_kernel", {})
        space_cfg = config.get("space_kernel", {})
        A = config.get("A", [[0.0]])
        
        # Instantiate time kernel using the provided type.
        ttype = time_cfg.get("type", "exponential")
        if ttype == "exponential":
            beta = time_cfg.get("beta", 1.0)
            time_kernel = ExponentialTimeKernel(beta)
        else:
            raise ValueError(f"Unknown time kernel type: {ttype}")
        
        # Instantiate space kernel using the provided type.
        stype = space_cfg.get("type", "gaussian")
        if stype == "gaussian":
            sigma = space_cfg.get("sigma", 1.0)
            space_kernel = GaussianSpaceKernel(sigma)
        else:
            raise ValueError(f"Unknown space kernel type: {stype}")
        
        return ParametricKernel(mode="separate", time_kernel=time_kernel, space_kernel=space_kernel, A=A)
    
    elif mode == "entangled":
        joint_cfg = config.get("joint_kernel", {})
        jtype = joint_cfg.get("type", "polynomial")
        if jtype == "polynomial":
            coeffs = joint_cfg.get("coefficients", {})
            joint_kernel = PolynomialJointKernel(coeffs)
        else:
            raise ValueError(f"Unknown joint kernel type: {jtype}")
        return ParametricKernel(mode="entangled", joint_kernel=joint_kernel)
    
    else:
        raise ValueError(f"Unknown kernel mode: {mode}")
