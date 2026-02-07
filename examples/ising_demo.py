"""
Simple Ising model demo - Pure Python/NumPy
"""

import numpy as np
import time
from thrml import SpinNode, IsingModel
from thrml.block_sampling import sample_states

def main():
    print("=== THRML-PY Ising Model Demo ===\n")
    
    # Parameters
    n = 20  # Grid size
    J = 2.5  # Strong coupling
    beta = 1.0
    
    print(f"Grid: {n}x{n} ({n*n} spins)")
    print(f"Coupling: J = {J}")
    print(f"Beta: β = {beta}\n")
    
    # Create nodes
    nodes = [SpinNode() for _ in range(n*n)]
    
    # Create grid edges
    edges = []
    for i in range(n):
        for j in range(n):
            idx = i * n + j
            if j < n-1:  # Right
                edges.append((idx, idx+1))
            if i < n-1:  # Down
                edges.append((idx, idx+n))
    
    print(f"Edges: {len(edges)}\n")
    
    # Couplings and fields
    weights = np.full(len(edges), J)
    biases = np.zeros(n*n)
    
    # Build model
    model = IsingModel(nodes, edges, biases, weights, beta)
    
    # Create sampling program
    print("Building sampling program...")
    program = model.create_sampling_program()
    print(f"Blocks: {len(program.gibbs_spec.free_blocks)}")
    print(f"Samplers: {len(program.samplers)}\n")
    
    # Initialize random state
    rng = np.random.default_rng(42)
    init_states = [rng.random(len(b)) < 0.5 for b in program.gibbs_spec.free_blocks]
    
    # Sample
    print("Sampling...")
    t0 = time.time()
    samples = sample_states(
        rng, program, init_states, [],
        n_warmup=200,
        n_samples=1000,
        thin=10
    )
    t1 = time.time()
    
    print(f"Time: {t1-t0:.2f}s\n")
    
    # Analyze results
    print("=== Results ===")
    
    all_samples = []
    for block_samples in samples:
        all_samples.append(block_samples)
    
    # Concatenate all blocks
    full_samples = np.concatenate(all_samples, axis=1)
    spins = full_samples.astype(float) * 2 - 1  # bool -> {-1, +1}
    
    # Magnetization per sample
    mag = np.mean(spins, axis=1)
    abs_mag = np.abs(mag)
    
    print(f"<M>   = {np.mean(mag):.4f} ± {np.std(mag):.4f}")
    print(f"<|M|> = {np.mean(abs_mag):.4f} ± {np.std(abs_mag):.4f}")
    print(f"Range: [{np.min(mag):.4f}, {np.max(mag):.4f}]")
    
    # Energy
    energies = np.array([model.energy(s) for s in full_samples])
    print(f"\n<E>   = {np.mean(energies):.2f} ± {np.std(energies):.2f}")
    
    print("\n✓ Demo complete!")

if __name__ == "__main__":
    main()
