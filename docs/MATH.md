# Mathematical Formulations and Derivations

This document outlines the primary mathematical models implemented in `echotrace-math`.

## 1. Bluestein Chirp-Z Transform
For arbitrary sequence length $N$, the DFT sum:
$$X[k] = \sum_{n=0}^{N-1} x[n] e^{-j 2\pi k n / N}$$
is rewritten using the identity $2kn = k^2 + n^2 - (k-n)^2$:
$$X[k] = e^{-j \pi k^2 / N} \sum_{n=0}^{N-1} \left( x[n] e^{-j \pi n^2 / N} \right) e^{j \pi (k-n)^2 / N}$$
This transforms the arbitrary DFT into a cyclic convolution evaluated via power-of-two FFT in $O(N \log N)$ operations.

## 2. Daubechies Wavelet Packets & Shannon Entropy
Orthogonal filter bank $h[n]$ and $g[n] = (-1)^n h[L-1-n]$.
Best-basis selection minimizes the additive information cost:
$$E(x) = -\sum_i p_i \ln p_i, \quad p_i = \frac{x_i^2}{\|x\|_2^2}$$
If $E(\text{parent}) \le E(\text{child}_1) + E(\text{child}_2)$, the node is pruned to the parent.

## 3. Schroeder Backward Integration & Lundeby Method
Continuous energy decay curve:
$$E(t) = \int_t^\infty h^2(\tau) d\tau$$
Lundeby truncation detects the intersection time $t_c$ where the decay envelope meets background noise $\sigma_n^2$, integrating over $[t, t_c]$ with noise floor bias subtraction:
$$E[k] = \sum_{j=k}^{t_c} \left( h^2[j] - \sigma_n^2 \right)$$
Linear regression slopes determine $T_{20}$ and $T_{30}$ reverberation times.

## 4. Peak-to-Correlation Energy (PCE)
Given test residual $W$ and reference sensor pattern $K$:
$$\text{PCE} = \frac{c(s_{\text{peak}})^2}{\frac{1}{|A| - |\mathcal{N}|} \sum_{s \notin \mathcal{N}} c(s)^2}$$
where $\mathcal{N}$ is a spatial neighborhood centered at the cross-correlation peak.

## 5. Bayesian Evidence Fusion
Log-odds formulation for hypothesis $H_1$ (tampered/leaking) vs $H_0$ (authentic):
$$\text{logit}(P(H_1 | E)) = \text{logit}(P(H_1)) + \sum_{i=1}^K w_i \ln(\text{LR}_i)$$
where $\text{LR}_i = P(E_i | H_1) / P(E_i | H_0)$ represents the likelihood ratio of forensic test $i$.

## 6. Multi-Head Attention with ALiBi
For head $h$ with slope $m_h = 2^{-8(h+1)/H}$:
$$S_{i,j}^h = \frac{q_i^T k_j}{\sqrt{d_k}} - m_h |i - j|$$
$$\text{Attn}_{i,j}^h = \text{softmax}_j(S_{i,j}^h)$$
$$O_i^h = \sum_j \text{Attn}_{i,j}^h v_j$$
