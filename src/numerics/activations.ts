/**
 * Numerically Stable Special Functions & Neural Activations
 *
 * Prevents underflow, overflow, and NaN across deep network layers.
 */

/**
 * Numerically stable sigmoid: sigma(x) = 1 / (1 + exp(-x))
 */
export function stableSigmoid(x: number): number {
  if (x >= 0) {
    const z = Math.exp(-x);
    return 1.0 / (1.0 + z);
  } else {
    const z = Math.exp(x);
    return z / (1.0 + z);
  }
}
export { stableSigmoid as sigmoid };

/**
 * Numerically stable log1p: log(1 + x)
 */
export function stableLog1p(x: number): number {
  return Math.log1p ? Math.log1p(x) : Math.log(1.0 + x);
}

/**
 * Numerically stable expm1: exp(x) - 1
 */
export function stableExpm1(x: number): number {
  return Math.expm1 ? Math.expm1(x) : Math.exp(x) - 1.0;
}

/**
 * Log-Sum-Exp of two numbers: log(exp(a) + exp(b))
 */
export function logSumExp2(a: number, b: number): number {
  const maxVal = Math.max(a, b);
  if (!isFinite(maxVal)) return maxVal;
  return maxVal + Math.log(Math.exp(a - maxVal) + Math.exp(b - maxVal));
}

/**
 * Log-Sum-Exp of an array: log(sum_i exp(x_i))
 */
export function logSumExp(values: ArrayLike<number>): number {
  const n = values.length;
  if (n === 0) return -Infinity;

  let maxVal = -Infinity;
  for (let i = 0; i < n; i++) {
    if (values[i]! > maxVal) maxVal = values[i]!;
  }
  if (!isFinite(maxVal)) return maxVal;

  let sum = 0.0;
  for (let i = 0; i < n; i++) {
    sum += Math.exp(values[i]! - maxVal);
  }

  return maxVal + Math.log(sum);
}

/**
 * In-place stable softmax: x_i -> exp(x_i - max) / sum(exp(x_j - max))
 */
export function stableSoftmax(input: ArrayLike<number>, output: Float32Array | Float64Array): void {
  const n = input.length;
  if (n === 0) return;

  let maxVal = -Infinity;
  for (let i = 0; i < n; i++) {
    if (input[i]! > maxVal) maxVal = input[i]!;
  }

  let sum = 0.0;
  for (let i = 0; i < n; i++) {
    const e = Math.exp(input[i]! - maxVal);
    output[i] = e;
    sum += e;
  }

  const invSum = 1.0 / Math.max(1e-12, sum);
  for (let i = 0; i < n; i++) {
    output[i] = output[i]! * invSum;
  }
}

/**
 * GELU (Gaussian Error Linear Unit): 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
 */
export function gelu(x: number): number {
  return 0.5 * x * (1.0 + Math.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)));
}

/**
 * Derivative of GELU for backpropagation: dGELU/dx
 */
export function geluDerivative(x: number): number {
  const k0 = 0.7978845608028654;
  const k1 = 0.044715;
  const poly = k0 * (x + k1 * x * x * x);
  const t = Math.tanh(poly);
  const dt_dx = (1.0 - t * t) * k0 * (1.0 + 3.0 * k1 * x * x);
  return 0.5 * (1.0 + t) + 0.5 * x * dt_dx;
}
