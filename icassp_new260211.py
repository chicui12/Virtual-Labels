"""
Strictly-proper-loss EM demo with partial labels
2-panel layout (risk + w-trajectory), one figure per loss.

This version:
- Runs only cross_entropy, brier, tsallis_0.5, spherical
- Q^k(w) dotted (uniform)
- Right-axis labels at w=4.0: Q1, Q2, Q3, finalQ
- NO title
- Bottom: smaller blue dots + LaTeX labels for w0..w3
- Vertical guide at w=4.0
- Saves EPS (vector) to ./loss_eps/
"""

import os
import re
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Vector font embedding for EPS
mpl.rcParams['ps.fonttype'] = 42

# ------------------------  global experiment settings  ------------------------
np.random.seed(42)
n          = 10000           # sample size
iterations = 50            # EM steps
w_star     = 2.0           # ground-truth weight
w0         = 3.0           # initial guess

# Range of parameter exploration
wmin  = min(w_star, w0) - 1
wmax  = max(w_star, w0) + 2            # = 4.0 with defaults
w_vals = np.linspace(wmin, wmax, 2001) # smooth but not too heavy

# ------------------------  utils  ------------------------
def slugify(s: str) -> str:
    return re.sub(r'[^A-Za-z0-9._-]+', '_', s).strip('_')

# ------------------------  data-generating helpers  ------------------------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def eta(x):
    """True posterior P(y=1|x) (binary ground truth) for scalar x."""
    fx = sigmoid(w_star * x)
    return np.array([1 - fx, fx])           # shape (2,)

def f_model(w, x_arr):
    """Model posterior at weight w for all x in x_arr."""
    fx = sigmoid(w * x_arr)
    return np.stack([1 - fx, fx], axis=1)   # shape (n,2)

def generate_data(n, M_mat):
    """
    Generate synthetic data with weak labels.
    M_mat: shape (num_z, 2) — rows weak labels, cols true labels
    """
    x      = np.random.uniform(-1, 1, n)
    y_true = np.array([np.random.choice(2, p=eta(xi)) for xi in x])
    z_obs  = np.array([np.random.choice(M_mat.shape[0], p=M_mat[:, y]) for y in y_true])
    return x, y_true, z_obs

# ------------------------  EM E-step ------------------------
def compute_q(f_curr, z, M):
    """
    q_i(y) ∝ P(z_i|y) f_curr_i(y)
    M: shape (num_z, 2) — rows weak labels, cols true labels
    """
    num = f_curr * M[z]         # broadcast (n,2)
    return num / num.sum(axis=1, keepdims=True)

# ------------------------  strictly-proper losses ------------------------
def get_loss_function(code):
    """
    Returns φ0(f), φ1(f) — loss for y=0 and y=1 as functions of f=P(y=1).
    code ∈ {cross_entropy, brier, spherical, tsallis_0.5}
    """
    eps = 1e-50

    if code == "cross_entropy":
        def phi(f):
            f = np.clip(f, eps, 1-eps)

            return -np.log(1-f), -np.log(f)

    elif code == "brier":
        def phi(f):
            return (0-f)**2, (1-f)**2

    elif code == "spherical":
        def phi(f):
            norm = np.sqrt((1-f)**2 + f**2)
            return -(1-f)/norm, -f/norm

    elif code == "ps_2":
        # def phi(f):
        #     beta = 2
        #     denom = (f**beta + (1-f)**beta)**(1.0 / beta)
        #     denom = np.maximum(denom, 1e-20)

        #     return -((1-f)**(beta) / (beta * denom), -(f**beta) / (beta * denom)

        def phi(f):
            beta = 2
            denom = (f**beta + (1-f)**beta)**((beta -1.0) / beta)
            denom = np.maximum(denom, 1e-20)

            return -((1-f)**(beta-1)) / denom, -(f**(beta-1)) / denom

    elif code.startswith("tsallis_"):
        alpha = float(code.rsplit("_", 1)[1])
        a = alpha - 1.0
        def phi(f):
            # Binary Tsallis proper scoring (scaled; scaling doesn't affect argmin)
            return (1 + a*((1-f)**alpha + f**alpha) - alpha*(1-f)**a)/a, \
                   (1 + a*(f**alpha + (1-f)**alpha) - alpha*f**a)/a

    else:
        raise ValueError(f"Unknown loss code {code}")

    return phi

# ------------------------  weak-label (partial-label) matrices  ------------------------
def col_norm(mat):
    """Normalise columns so each is a conditional distribution P(z|y)."""
    return mat / mat.sum(axis=0, keepdims=True)

M_variants = {
    "2×2 (asym noise)": col_norm(np.array([
        [0.80, 0.05],
        [0.20, 0.95],
    ])),
}

# ------------------------  run one EM chain  ------------------------
def run_chain(loss_code, M_mat, x, z_obs, w_vals):
    phi = get_loss_function(loss_code)

    # ---- EM loop ----
    w_curr    = w0
    w_history = [w_curr]
    Q_curves  = []

    for _ in range(iterations):
        f_curr = f_model(w_curr, x)
        q      = compute_q(f_curr, z_obs, M_mat)          # shape (n,2)

        # Compute the weak-label risk at the current w (for alignment)
        f_k  = f_model(w_curr, x)
        Mfk  = (M_mat @ f_k.T).T
        phi0, phi1 = phi(Mfk[:, 1])
        R_curr = np.sum(np.where(z_obs==1, phi1, phi0))

        # Compute Q at the current w
        phi0, phi1 = phi(f_k[:, 1])
        Q_curr = np.sum(q[:, 0]*phi0 + q[:, 1]*phi1)

        # --- M-step: minimise expected loss Q(w) over the grid ---
        Q_vals = []
        for w in w_vals:
            f_kw         = f_model(w, x)
            phi0w, phi1w = phi(f_kw[:, 1])
            # Align levels so Q^k and R coincide at current w (for display)
            Qw = np.sum(q[:, 0]*phi0w + q[:, 1]*phi1w) + R_curr - Q_curr
            Q_vals.append(Qw)
        Q_vals = np.array(Q_vals)
        Q_curves.append(Q_vals)

        w_curr = w_vals[int(np.argmin(Q_vals))]
        w_history.append(w_curr)

    # ---- risk R(w) under weak labels z ----
    R_vals = []
    for w in w_vals:
        f_kw  = f_model(w, x)
        Mfkw  = (M_mat @ f_kw.T).T
        phi0w, phi1w = phi(Mfkw[:, 1])
        R_vals.append(np.sum(np.where(z_obs==1, phi1w, phi0w)))

    return Q_curves, np.array(R_vals), w_history

# ------------------------  plotting helper (2 panels)  ------------------------
def plot_one_loss(loss_code, M_name, M_mat, x, z_obs, save_dir=None):
    Qs, R, w_hist = run_chain(loss_code, M_mat, x, z_obs, w_vals)

    fig, axes = plt.subplots(1, 2, figsize=(6, 5),
                             gridspec_kw={"width_ratios": [3, 1]})

    # === TOP: risk landscape ===
    ax = axes[0]

    # Styles (uniform dotted Q, distinct solid R)
    r_style = dict(linestyle='-', linewidth=2.0)
    q_style = dict(linestyle=':', linewidth=1.4, alpha=0.7)

    # Plot R(w)
    ax.plot(w_vals, R, **r_style, label=r'True risk $R(w)$')

    # Plot Q^k(w) and collect line handles
    q_lines = []
    for Q in Qs:
        (ln,) = ax.plot(w_vals, Q, **q_style)
        q_lines.append(ln)

    # Legend: one entry for R and one for "all Q"
    handles = [
        Line2D([0], [0], **r_style, 
               label=r'True risk $R(w)$' + f' ({loss_code.replace("_", " ")})   '),
        Line2D([0], [0], **q_style, label=r'$Q_k(w)$ (dotted)'),
    ]
    ax.legend(handles=handles, loc="best", frameon=False, title="Curves")

    # Right-axis labels at w = 4.0 (Q1, Q2, Q3, finalQ)
    idx_right = -1  # last w = wmax
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())

    # pick first up to 3 iterations + final
    idx_picks = list(range(min(3, len(Qs))))  # 0,1,2 if available
    labels    = [f"$Q_{i+1}$" for i in idx_picks]
    if len(Qs) > 0:
        idx_picks.append(len(Qs) - 1)
        labels.append("$Q_{50}$")

    # collect tick positions; deduplicate while preserving order
    yvals_raw = [Qs[i][idx_right] for i in idx_picks]
    seen = set()
    yvals, ylabels = [], []
    for y, lab in zip(yvals_raw, labels):
        key = round(float(y), 10)  # guard against fp jitter
        if key not in seen:
            seen.add(key)
            yvals.append(y)
            ylabels.append(lab)

    # set ticks if within y-limits
    if yvals:
        ylow, yhigh = ax.get_ylim()
        inrange = [(y, lab) for y, lab in zip(yvals, ylabels) if ylow <= y <= yhigh]
        if inrange:
            ax2.set_yticks([y for y, _ in inrange])
            ax2.set_yticklabels([lab for _, lab in inrange])

    ax2.tick_params(axis='y', which='both', length=4)
    ax2.grid(False)

    # Visual guide at the right edge (w = wmax, typically 4.0)
    ax.axvline(w_vals[idx_right], linestyle='--', linewidth=1.0, alpha=0.5)

    ax.set_ylabel("Empirical risk")
    ax.set_xlim(w_vals[0], w_vals[-1])
    # NOTE: no title (as requested)

    # === BOTTOM: w trajectory ===
    iters = np.arange(len(w_hist))
    axes[1].plot(iters, w_hist, marker='o', markersize=3, linewidth=1.6, label=r'$w_k$')
    for k in range(min(4, len(w_hist))):
        axes[1].scatter(k, w_hist[k], s=16, zorder=3)
        axes[1].annotate(rf"$w_{{{k}}}$",
                         xy=(k, w_hist[k]),
                         xytext=(k + 0.35, w_hist[k]),
                         textcoords="data",
                         fontsize=11, ha="left", va="center")

    # risk minimizer under R(w) and ground truth
    i_min = int(np.argmin(R))
    w_min = w_vals[i_min]
    axes[1].axhline(w_star, ls='--', label=r'$w^*$')
    axes[1].axhline(w_min,  ls=':',  label=r'$w_{\min}$')

    axes[1].set_xlabel("Iteration, k")
    # axes[1].set_ylabel(r"$w$")
    axes[1].legend()
    axes[1].set_xlim(0, len(w_hist)-1)

    fig.tight_layout()

    # --- SAVE: EPS only, with alpha fix for EPS ---
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        safe_loss = loss_code.replace(".", "_")
        safe_M    = slugify(M_name)
        base      = os.path.join(save_dir, f"{safe_loss}_{safe_M}")

        # Temporarily remove transparency from Q lines (EPS dislikes alpha)
        old_alphas = [ln.get_alpha() for ln in q_lines]
        for ln in q_lines:
            if ln.get_alpha() is not None and ln.get_alpha() < 1.0:
                ln.set_alpha(1.0)

        fig.savefig(base + ".png", bbox_inches="tight", dpi=300)

        # Restore original alphas
        for ln, a in zip(q_lines, old_alphas):
            ln.set_alpha(a)

        print(f"Saved: {base}.eps")

    plt.show()
    # plt.close(fig)  # avoid figure pile-up when looping

# ------------------------  main: generate selected losses  ------------------------
def main(save_dir=None):
    losses = ["ps_2", "cross_entropy", "brier", "tsallis_0.5", "spherical"]

    for M_name, M_mat in M_variants.items():
        x, y_true, z_obs = generate_data(n, M_mat)
        for loss_code in losses:
            plot_one_loss(loss_code, M_name, M_mat, x, z_obs, save_dir=save_dir)

if __name__ == "__main__":
    # Saves EPS to ./loss_eps/
    main(save_dir="loss_eps")
