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

## 7. Electric Network Frequency (ENF) Harmonic Fusion & CRLB
The Cramer-Rao Lower Bound (CRLB) for the variance of an unbiased frequency estimator $\hat{f}$ of a sinusoid in white Gaussian noise (Rife & Boorstyn 1974):
$$\text{Var}(\hat{f}) \ge \frac{12}{(2\pi)^2 \cdot \text{SNR} \cdot T_s^2 \cdot N(N^2 - 1)}$$
For multiharmonic grid signals with harmonic orders $m \in \{1, 2, 3, 4\}$, weighted maximum ratio combining fuses the estimated frequencies $\hat{f}_m$:
$$\hat{f}_{\text{grid}} = \frac{\sum_{m} w_m \cdot (\hat{f}_m / m)}{\sum_m w_m}, \quad w_m = \text{SNR}_m \cdot m^2$$
This achieves the joint CRLB over all observable grid harmonics.

## 8. Subspace Direction of Arrival (MUSIC & ESPRIT)
Given spatial covariance matrix $R_{xx} = E[x(t) x^H(t)] = E_s \Lambda_s E_s^H + E_n \Lambda_n E_n^H$:
1. **MUSIC Pseudo-Spectrum (Schmidt 1986):**
   $$P_{\text{MUSIC}}(\theta) = \frac{1}{a^H(\theta) E_n E_n^H a(\theta)}$$
2. **ESPRIT Rotational Invariance (Roy & Kailath 1989):**
   Subarray partitioning yields $E_{s1} \Psi = E_{s2}$. The eigenvalues $\phi_k$ of $\Psi = (E_{s1}^H E_{s1})^{-1} E_{s1}^H E_{s2}$ directly solve the DOA angles:
   $$\theta_k = \arcsin\left(\frac{\arg(\phi_k)}{\pi}\right)$$

## 9. Remote Photoplethysmography (CHROM & POS)
Hemodynamic blood volume pulse extraction from RGB facial skin chrominance signals (de Haan & Jeanne 2013, Wang et al. 2017):
1. **CHROM (Chrominance-based):**
   $$X_s = 3 R - 2 G$$
   $$Y_s = 1.5 R + G - 1.5 B$$
   $$\text{BVP} = X_s - \alpha Y_s, \quad \alpha = \frac{\sigma(X_s)}{\sigma(Y_s)}$$
2. **POS (Plane-Orthogonal-to-Skin):**
   Projects normalized RGB signals onto the temporal pulsatile plane orthogonal to skin tone drift, bandpassing within $[0.7, 4.0]\text{ Hz}$ (42 to 240 BPM).

## 10. Robust Estimation & False Discovery Rate Control
1. **Huber Loss M-Estimator:**
   $$\rho_\delta(r) = \begin{cases} \frac{1}{2} r^2 & |r| \le \delta \\ \delta(|r| - \frac{1}{2}\delta) & |r| > \delta \end{cases}$$
2. **Benjamini-Hochberg False Discovery Rate (FDR):**
   For sorted $p$-values $p_{(1)} \le p_{(2)} \le \dots \le p_{(M)}$:
   $$k = \max \left\{ i : p_{(i)} \le \frac{i}{M} Q \right\}$$
   Rejects null hypotheses $H_{(1)}, \dots, H_{(k)}$ guaranteeing $\text{FDR} \le Q$.

## 11. Dual-Method Cross-Checks
- **Bluestein FFT vs Naive DFT:** Pointwise error $< 10^{-10}$ across arbitrary prime lengths $N$.
- **Welch vs DPSS Multitaper:** Spectral peak agreement within single-bin resolution ($\Delta f \le 2.73\text{ Hz}$).
- **MUSIC vs ESPRIT:** Subspace eigenvalue concordance for spatial angle tracking.

## 12. Automated Detector Parameter Tuning
Nelder-Mead simplex and golden-section optimization on validation sets lock detection thresholds:
- Beacon CFAR SNR: 14.0 dB
- Welch Segment Length: 1024 samples
- CUSUM Splice Sensitivity $h$: 3.0
