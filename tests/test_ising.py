"""
Unit tests for THRML-PY Ising model
"""

import numpy as np
from thrml import SpinNode, IsingModel
from thrml.block_sampling import sample_states


def test_ising_basic():
    """Test basic Ising model functionality"""
    
    # Small 4-node chain
    nodes = [SpinNode() for _ in range(4)]
    edges = [(0, 1), (1, 2), (2, 3)]
    weights = np.ones(3)
    
    model = IsingModel(nodes, edges, weights=weights, beta=1.0)
    program = model.create_sampling_program()
    
    assert len(program.gibbs_spec.free_blocks) == 4
    assert len(program.samplers) == 4


def test_ising_sampling():
    """Test sampling produces valid states"""
    
    nodes = [SpinNode() for _ in range(10)]
    edges = [(i, i+1) for i in range(9)]
    weights = np.ones(9)
    
    model = IsingModel(nodes, edges, weights=weights, beta=1.0)
    program = model.create_sampling_program()
    
    rng = np.random.default_rng(42)
    init_states = [rng.random(len(b)) < 0.5 for b in program.gibbs_spec.free_blocks]
    
    samples = sample_states(
        rng, program, init_states, [],
        n_warmup=10, n_samples=50, thin=1
    )
    
    # Check output shape
    for block_idx, block_samples in enumerate(samples):
        expected_shape = (50, len(program.gibbs_spec.free_blocks[block_idx]))
        assert block_samples.shape == expected_shape
        assert block_samples.dtype == bool


def test_ising_magnetization():
    """Test magnetization calculation"""
    
    # 5x5 grid with strong ferromagnetic coupling
    n = 5
    nodes = [SpinNode() for _ in range(n*n)]
    
    edges = []
    for i in range(n):
        for j in range(n):
            idx = i * n + j
            if j < n-1:
                edges.append((idx, idx+1))
            if i < n-1:
                edges.append((idx, idx+n))
    
    J = 2.0
    weights = np.full(len(edges), J)
    
    model = IsingModel(nodes, edges, weights=weights, beta=1.0)
    program = model.create_sampling_program()
    
    rng = np.random.default_rng(123)
    init_states = [rng.random(len(b)) < 0.5 for b in program.gibbs_spec.free_blocks]
    
    samples = sample_states(
        rng, program, init_states, [],
        n_warmup=100, n_samples=200, thin=5
    )
    
    # Compute magnetization
    all_samples = np.concatenate(samples, axis=1)
    spins = all_samples.astype(float) * 2 - 1
    mag = np.mean(spins, axis=1)
    abs_mag = np.mean(np.abs(mag))
    
    # Strong coupling should give high |M|
    print(f"<|M|> = {abs_mag:.3f}")
    assert abs_mag > 0.5, f"Expected high magnetization, got {abs_mag:.3f}"


def test_energy_calculation():
    """Test energy function"""
    
    nodes = [SpinNode() for _ in range(3)]
    edges = [(0, 1), (1, 2)]
    weights = np.array([1.0, 1.0])
    biases = np.array([0.5, 0.0, -0.5])
    
    model = IsingModel(nodes, edges, biases=biases, weights=weights, beta=1.0)
    
    # All spins up
    state_up = np.array([True, True, True])
    E_up = model.energy(state_up)
    
    # All spins down
    state_down = np.array([False, False, False])
    E_down = model.energy(state_down)
    
    # Due to biases, energies should differ
    assert E_up != E_down
    print(f"E(↑↑↑) = {E_up:.3f}")
    print(f"E(↓↓↓) = {E_down:.3f}")


if __name__ == "__main__":
    print("Running tests...\n")
    
    test_ising_basic()
    print("✓ test_ising_basic")
    
    test_ising_sampling()
    print("✓ test_ising_sampling")
    
    test_ising_magnetization()
    print("✓ test_ising_magnetization")
    
    test_energy_calculation()
    print("✓ test_energy_calculation")
    
    print("\n✓ All tests passed!")
