## Additional Experiments (MNIST)

This section summarizes additional MNIST results comparing two training strategies under the same weak-label model:

- Marginal Chain (MC)
- ForwardProperLoss (FWD): direct optimization of the corresponding forward-corrected / M-proper objective

### Key observations

- Base loss = cross-entropy  
  MC and FWD behave identically in practice, producing the same learning curves and the same final accuracy (up to minor run-to-run noise).

- Base loss ≠ cross-entropy (e.g., Brier, Tsallis with α = 0.2, pseudo-spherical with p = 2)  
  MC and FWD no longer coincide and can converge to different solutions with different accuracies.

### Representative example: pseudo-spherical (p = 2)

Pseudo-spherical loss with p = 2 provides a clear and typical case where the methods diverge:

- MC plateaus below 0.5 test accuracy
- FWD reach close to 0.8 test accuracy under the same setting

This example illustrates that MC matches direct optimization only in the cross-entropy case, while for other proper losses MC can converge to a biased fixed point.

### EXperiment Results table（More experimental figures are available in the figs and Experiment Results directory on GitHub.）


| Base loss | MC (test acc) | FWD (test acc)
|---|---:|---:
| Pseudo-spherical (p = 2) | 0.4980 （<0.5） | 0.7603 (~0.8) |
| Cross-entropy | 0.9299 | 0.9299 |
| Brier | 0.9309 | 0.9225 | 
| Tsallis (α = 0.2) | 0.8974 | 0.8991 | 


### Experiment Results Figures

**Figure 1.** Pseudo-Spherical (p=2): MC vs FWD
![MNIST pseudo-spherical p=2](figs/ps_2_mc_vs_fwd.png)


**Figure 2.** Cross Entropy: MC vs FWD
![FWD CE](figs/cross_entropy_mc_vs_fwd.png)

**Figure 3.** Brier: MC vs FWD
![FWD CE](figs/brier_mc_vs_fwd.png)

**Figure 4.** Tsallis0.2: MC vs FWD
![FWD CE](figs/tsallis_0.2_mc_vs_fwd.png)
