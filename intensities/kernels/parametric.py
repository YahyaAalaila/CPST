# File: stdgp/kernels/modular_kernel.py
class ParametricKernel:
    """
    ModularKernel provides a unified interface for the triggering kernel and 
    can operate in two modes:
    
      - 'separate': uses a product of a time kernel and a space kernel.
      - 'entangled': uses a joint kernel that directly models the spatiotemporal interaction.
    """
    def __init__(self, mode: str, time_kernel=None, space_kernel=None, joint_kernel=None, A=None):
        """
        Parameters:
            mode: 'separate' or 'entangled'
            time_kernel: instance of BaseTimeKernel (required if mode == 'separate')
            space_kernel: instance of BaseSpaceKernel (required if mode == 'separate')
            joint_kernel: instance of JointKernel (required if mode == 'entangled')
            A: numpy array for branching matrix (required in separate mode),
               so that A[candidate_type-1, past_type-1] gives the branching factor.
        """
        if mode not in ['separate', 'entangled']:
            raise ValueError("mode must be 'separate' or 'entangled'")
        self.mode = mode
        
        if self.mode == 'separate':
            if time_kernel is None or space_kernel is None or A is None:
                raise ValueError("For separate mode, time_kernel, space_kernel, and A must be provided.")
            self.time_kernel = time_kernel
            self.space_kernel = space_kernel
            self.A = A
        else:  # entangled
            if joint_kernel is None:
                raise ValueError("For entangled mode, joint_kernel must be provided.")
            self.joint_kernel = joint_kernel

    def evaluate(self, dt: float, dx: float, dy: float) -> float:
        """
        Evaluate the kernel contribution from a past event at (t_i, x_i, y_i, past_type)
        to the candidate event at (t, x, y, candidate_type).
        """
        if dt < 0:
            return 0.0
        if self.mode == 'separate':
            time_val = self.time_kernel.evaluate(dt)
            space_val = self.space_kernel.evaluate(dx, dy)
            return time_val * space_val
        else:  # entangled
            return self.joint_kernel.evaluate(dt, dx, dy)
