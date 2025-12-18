## Additional Experiments (MNIST)

This section summarizes additional MNIST results comparing three training strategies under the same weak-label model:

- Marginal Chain (MC)
- ForwardProperLoss (FWD): direct optimization of the corresponding forward-corrected / M-proper objective
- UpperBound (MM): optimization of the derived majorization (upper-bound) objective

### Key observations

- Base loss = cross-entropy  
  MC, FWD, and UpperBound behave identically in practice, producing the same learning curves and the same final accuracy (up to minor run-to-run noise).

- Base loss ≠ cross-entropy (e.g., Brier, Tsallis with α = 0.2, pseudo-spherical with p = 2)  
  MC, FWD, and UpperBound no longer coincide and can converge to different solutions with different accuracies.

### Representative example: pseudo-spherical (p = 2)

Pseudo-spherical loss with p = 2 provides a clear and typical case where the methods diverge:

- MC plateaus below 0.5 test accuracy
- FWD and UpperBound both reach close to 0.8 test accuracy under the same setting

This example illustrates that MC matches direct optimization only in the cross-entropy case, while for other proper losses MC can converge to a biased fixed point.

### Results table (placeholder)

Fill in the numbers below with your runs (recommended: report mean ± std over several random seeds).

| Base loss | MC (test acc) | FWD (test acc) | UpperBound (test acc) |
|---|---:|---:|---:|
| Cross-entropy | 0.9299 | 0.9299 | 0.9299 |
| Brier | 0.9309 | 0.9225 | 0.9221 |
| Tsallis (α = 0.2) | 0.8974 | 0.8991 | 0.8996 |
| Pseudo-spherical (p = 2) | 0.4980 （<0.5） | 0.7603 (~0.8) | 0.7594(~ 0.8) |

### Figures (placeholders)

Add your plots to `results/figs/` (or any folder you prefer) and update the paths below.

Pseudo-spherical (p = 2) training curves:
![MNIST pseudo-spherical p=2](results/figs/mnist_pseudo_spherical_p2_curves.png)





Optional: overall comparison across base losses:

![MNIST comparison across base losses](results/figs/mnist_all_losses_summary.png)

Optional: tables/plots for multiple seeds:

![MNIST multi-seed summary](results/figs/mnist_multiseed_summary.png)
