import numpy as np

def thinning(background, kernel, time_horizon: float, domain: list, A: np.array, Lambda: float, rng=None):
    """
    Thinning algorithm to generate spatiotemporal events.
    
    Parameters:
        composite_intensity: instance of BaseIntensity (e.g. CompositeIntensity).
        time_horizon: float, maximum simulation time.
        domain: list, e.g. [x_min, x_max, y_min, y_max].
        m: int, number of marks.
        Lambda: float, global upper bound on intensity.
        rng: np.random.Generator, defaults to np.random.default_rng().
    
    Returns:
        events: list of dicts with keys: 't', 'x', 'y', 'type', 'lambda'
    """
    m = A.shape[0]  # Number of marks/types
    if len(domain) != 4:
        raise ValueError("Domain must be a list of four elements: [x_min, x_max, y_min, y_max].")
    if rng is None:
        rng = np.random.default_rng()
        
    events = []
    t_current = 0.0
    while t_current < time_horizon:
        dt = -np.log(rng.uniform()) / Lambda
        t_candidate = t_current + dt
        if t_candidate > time_horizon:
            break
        x_candidate = rng.uniform(domain[0], domain[1])
        y_candidate = rng.uniform(domain[2], domain[3])
        candidate_type = rng.integers(1, m+1)
        lam_val = background.evaluate(t_candidate, x_candidate, y_candidate)
        trig = 0.0
        for e in events:
            if e['t'] < t_candidate:
                dt_candidate = t_candidate - e['t']
                dx = x_candidate - e['x']
                dy = y_candidate - e['y']
                trig += A[candidate_type - 1, e['type'] - 1] * kernel.evaluate(dt_candidate, dx, dy)

        lambda_cand = lam_val + trig
        if rng.uniform() < lambda_cand / Lambda:
            events.append({
                't': t_candidate,
                'x': x_candidate,
                'y': y_candidate,
                'type': candidate_type,
                'lambda': lambda_cand
            })
        t_current = t_candidate
    events.sort(key=lambda e: e['t'])
    return events
