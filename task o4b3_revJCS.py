"""
Strictly-proper-loss EM demo with partial labels
Author: <you>
Requires: numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt

# ------------------------  global experiment settings  ------------------------
np.random.seed(42)
n          = 100           # sample size
iterations = 50                  # EM steps
w_star     = 2.0                   # ground-truth weight
w0         = 3                     # initial guess

# Range of parameter exploration
wmin = min(w_star, w0) - 1
wmax = max(w_star, w0) + 1
w_vals   = np.linspace(wmin, wmax, 10001)  # wide enough to see full curves


# ------------------------  data-generating helpers  ------------------------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def eta(x):
    """True posterior P(y=1|x) (binary ground truth)."""
    fx = sigmoid(w_star * x)
    return np.array([1 - fx, fx])           # shape (n,2)
    # return np.stack([1 - fx, fx], axis=1)           # shape (n,2)

def f_model(w, x):
    """Model posterior at weight w."""
    fx = sigmoid(w * x)
    return np.stack([1 - fx, fx], axis=1)

def generate_data(n, M_mat):
    """
    Generate synthetic data with weak labels.
    M_mat: shape (num_z, 2) — rows weak labels, cols true labels

    [Data were generated inside run_chain(), but it is better to do it here,
     and call this method only once, to avoid recomputing the same data and
     so that we can use the same data for all loss functions.]
    """

    np.random.seed(42)
    x      = np.random.uniform(-1, 1, n)
    y_true = np.array([np.random.choice(2, p=eta(xi)) for xi in x])
    z_obs  = np.array([np.random.choice(M_mat.shape[0], p=M_mat[:, y])
                       for y in y_true])
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
    code ∈ {cross_entropy, brier, spherical,
            tsallis_0.5, tsallis_2,
            ps_0.5, ps_3}
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

    elif code.startswith("tsallis_"):
        alpha = float(code.rsplit("_", 1)[1])
        a = alpha - 1
        def phi(f):
            return (1+a*((1-f)**alpha+f**alpha)-alpha*(1-f)**a)/a,(1+a*(f**alpha+(1-f)**alpha)-alpha*f**a)/a

    elif code.startswith("ps_"):        # pseudo-spherical
        beta = float(code.split("_")[1])
        def phi(f):
            den = ((1-f)**beta + f**beta)**(1/beta)


            #a1 = (- f**beta) / beta
            #a0 = (- (1-f)**beta) / beta
            # a0 = (1-f)**beta
            # a1 = f**betas
            # denom = (a0 + a1)**(1/beta)
            # return -a0/denom, -a1/denom
            #return a0, a1
            return -((1-f)**beta)/(beta*den), -(f**beta)/(beta*den)

    else:
        raise ValueError(f"Unknown loss code {code}")

    return phi


# ------------------------  weak-label (partial-label) matrices  ------------------------
def col_norm(mat):
    """Normalise columns so each is a conditional distribution P(z|y)."""
    return mat / mat.sum(axis=0, keepdims=True)

M_variants = {
    # "2×2 (sym noise)": col_norm(np.array([
    #     [0.90, 0.1],
    #     [0.10, 0.9],
    #])),

    "2×2 (assym noise)": col_norm(np.array([
        [0.80, 0.05],
        [0.20, 0.95],
    ])),
}

# Non-square matrices cannot be used in the current version of the code
# They require a multiclass loss function, to be implemented later
#     "3×2": col_norm(np.array([
#         [0.80, 0.10],
#         [0.15, 0.60],
#         [0.05, 0.30],
#     ])),
#     "4×2": col_norm(np.array([
#         [0.70, 0.20],
#         [0.20, 0.50],
#         [0.07, 0.25],
#         [0.03, 0.05],
#     ])),

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

        # Compute the true risk at the current w
        f_k        = f_model(w_curr, x)
        Mfk = (M_mat @ f_k.T).T
        phi0, phi1 = phi(Mfk[:, 1])
        R_curr = np.sum(np.where(z_obs==1, phi1, phi0))
        # Compute Q at the current w
        phi0, phi1    = phi(f_k[:, 1])
        Q_curr = np.sum(q[:, 0]*phi0 + q[:, 1]*phi1)

        # --- M-step: minimise expected loss Q(w) over the grid ---
        Q_vals = []
        for w in w_vals:
            f_k           = f_model(w, x)
            phi0, phi1    = phi(f_k[:, 1])
            Qw = np.sum(q[:, 0]*phi0 + q[:, 1]*phi1) + R_curr - Q_curr

            Q_vals.append(Qw)
        Q_curves.append(Q_vals)

        w_curr = w_vals[int(np.argmin(Q_vals))]
        w_history.append(w_curr)

    # ---- true risk R(w) under the real labels ----
    R_vals = []
    for w in w_vals:
        f_k        = f_model(w, x)
        # phi0, phi1 = phi(f_k[:, 1])
        # R_vals.append(np.sum(np.where(y_true==1, phi1, phi0)))
        # Compute risk R(w) = sum_k{zk'·φ(M·fk)}
        Mfk = (M_mat @ f_k.T).T
        phi0, phi1 = phi(Mfk[:, 1])
        R_vals.append(np.sum(np.where(z_obs==1, phi1, phi0)))

    return Q_curves, R_vals, w_history


# ------------------------  run everything and plot  ------------------------
# losses = ["cross_entropy", "brier", "spherical",
#           "tsallis_0.5", "tsallis_2",
#           "ps_0.5", "ps_3"]
losses = ["cross_entropy", "brier", "spherical", "tsallis_0.5", "tsallis_2",
         ]
# losses = ["tsallis_10", "tsallis_4"]
# Generate data once

for M_name, M_mat in M_variants.items():

    print(f"-- -- Transition matrix {M_name}")
    # Generate data
    x, y_true, z_obs = generate_data(n, M_mat)

    for loss_code in losses:
        print(f"-- Exploring loss {loss_code}...")

        # Run the chain
        Qs, R, w_hist = run_chain(loss_code, M_mat, x, z_obs, w_vals)

        fig, axes = plt.subplots(2, 1, figsize=(10, 8),
                                 gridspec_kw={"height_ratios": [3, 1]})

        # --- top: risk landscape ---
        ax = axes[0]
        ax.plot(w_vals, R, 'r-', linewidth=2, label='True risk $R(w)$')
        for i, Q in enumerate(Qs):
            ax.plot(w_vals, Q, alpha=0.5, label=f'$Q^{{{i}}}(w)$')
            # mark current w on that Q curve
            ax.scatter(w_hist[i], Q[int(np.argmin(np.abs(w_vals - w_hist[i])))],
                       color='k', s=25)
        ax.set_ylabel("Empiricalisk")
        ax.set_title(f"{loss_code}  —  weak-label matrix: {M_name}")
        if len(Qs) <= 15:
            ax.legend()
        # Fit the bounds of the y-axis to the data
        # ax.set_ylim(bottom=0)
        ax.set_xlim(w_vals[0], w_vals[-1])


        # --- risk minimizer
        i_min = np.argmin(R)
        w_min = w_vals[i_min]

        # --- bottom: w trajectory ---
        axes[1].plot(range(len(w_hist)), w_hist, marker='o')
        axes[1].axhline(w_star, ls='--', color='gray', label='$w^*$')
        axes[1].axhline(w_min, ls=':', color='gray', label='$w_{\min}$')
        axes[1].set_xlabel("Iteration")
        axes[1].set_ylabel("$w$")
        axes[1].legend()

        fig.tight_layout()
        # plt.savefig(f"./Code/figs/fig_{loss_code}_{M_name}.png")
plt.show()