"""Generate a random direction at layer 26 matched in norm to the
real diff-of-means direction. This is the control vector for the
random-direction baseline experiment.

We want a direction that is:
  - Same dimensionality (3584, hidden_dim of Qwen-7B)
  - Same L2 norm (4.107, matching direction_layer26_round_4.npy)
  - Sampled from an isotropic Gaussian then normalized

We save it to disk and verify it has near-zero cosine similarity with
the real direction. If cos > 0.05 we resample.
"""

from pathlib import Path

import numpy as np

DIR_DIR = Path("outputs/direction_results")
REAL = DIR_DIR / "direction_layer26_round_4.npy"
OUT = DIR_DIR / "direction_layer26_random_matched.npy"

assert REAL.exists(), f"Real direction not found at {REAL}"

real = np.load(REAL)
real_norm = np.linalg.norm(real)
print(f"Real direction: shape={real.shape}, ||d||={real_norm:.4f}")

rng = np.random.default_rng(seed=42)

for attempt in range(20):
    # Sample isotropic Gaussian, normalize to match real direction's norm
    rand = rng.standard_normal(real.shape)
    rand = rand / np.linalg.norm(rand) * real_norm

    cos_sim = np.dot(rand, real) / (np.linalg.norm(rand) * np.linalg.norm(real))
    print(f"  Attempt {attempt}: ||rand||={np.linalg.norm(rand):.4f}, "
          f"cos(rand, real)={cos_sim:+.4f}")

    if abs(cos_sim) < 0.05:
        break
else:
    raise RuntimeError("Could not find a random direction with |cos| < 0.05")

np.save(OUT, rand)
print(f"\nSaved random direction → {OUT}")
print(f"Final: ||rand||={np.linalg.norm(rand):.4f}, "
      f"cos(rand, real)={cos_sim:+.4f}")
print("\nThis vector is matched in:")
print(f"  - shape: {rand.shape}")
print(f"  - norm:  {np.linalg.norm(rand):.4f} (real: {real_norm:.4f})")
print(f"  - dimensionality: same residual stream layer (26)")
print(f"\nIt differs from the real direction in:")
print(f"  - direction (cos similarity is near zero — orthogonal)")
print("\nIf steering with this random vector produces a pos-vs-neg contrast")
print("similar to v1's, then v1's effect is 'any nontrivial perturbation at")
print("layer 26 disrupts deliberation' — i.e., not direction-specific.")
print("If random produces no contrast, v1's direction is doing something")
print("specific to the internalization feature.")