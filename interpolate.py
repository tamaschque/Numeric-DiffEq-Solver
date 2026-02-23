import numpy as np

def map_range(x, x_min, x_max, y_min, y_max):
    # Ensure x is in given range
    x = max(min(x, x_max), x_min)

    x_rel = (x - x_min) / (x_max - x_min)

    return (y_max - y_min) * x_rel + y_min

def interpolate_points(t, t_values, y_values):
    """Interpolates the given y values of a function y(t)."""
    t_values = np.array(t_values)
    y_values = np.array(y_values)

    if np.any(t > np.max(t_values)):
        return y_values[-1]

    if t in t_values:
        idx = t_values.tolist().index(t)
        return y_values[idx]
    else:
        t_low = t_values[t_values < t][-1]
        t_high = t_values[t_values > t][0]

        y_low = y_values[t_values.tolist().index(t_low)]
        y_high = y_values[t_values.tolist().index(t_high)]

        return map_range(t, t_low, t_high, y_low, y_high)

def interpolate_points2d(t, x, t_values, x_values, y_values):
    """Interpolates the given y values of a function y(t,x). y_values should be of shape (len_t, len_x)."""
    fx = interpolate_points(t, t_values, y_values)

    return interpolate_points(x, x_values, fx)

