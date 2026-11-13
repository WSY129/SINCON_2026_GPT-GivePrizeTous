"""Second look at the stack: weights that differ by type of day (and optionally by block of hours).

The v12 pipeline gives every day the same member weights.  Here the pipeline output `base` is moved towards / away from
single members:  new = base + sum_m delta[g, m] * (member_m - base),  one delta per group g of cells, ridge-shrunk to zero
(delta = 0 is the pipeline itself).  Fitted nested and date-blocked like every other correction.
"""
import numpy as np

HOUR_BLOCKS = {'all': [(0, 18)], 'four': [(0, 4), (4, 10), (10, 14), (14, 18)]}      # 06-09, 10-15, 16-19, 20-23


def design(base, members, use):
    """[cells, M] differences member - base for the cells in `use` ([day, S, H] boolean)."""
    return np.stack([m[use] - base[use] for m in members], 1)


def fit(parts, n_groups, ridge):
    """parts: list of (D [n, M], residual [n], group [n]).  Returns delta [n_groups, M]."""
    M = parts[0][0].shape[1]; out = np.zeros((n_groups, M))
    for g in range(n_groups):
        A = np.zeros((M, M)); b = np.zeros(M); n = 0
        for D, r, grp in parts:
            m = grp == g
            if m.any():
                A += D[m].T @ D[m]; b += D[m].T @ r[m]; n += int(m.sum())
        if n >= 2000:
            out[g] = np.linalg.solve(A + ridge * n * np.eye(M), b)
    return out


def apply(base, members, use, group, delta):
    out = base.copy(); D = design(base, members, use)
    out[use] = np.maximum(base[use] + (D * delta[group[use]]).sum(1), 0)
    return out
