# THRML Python Implementation

### Thermodynamic Hypergraphical Model Library Port

A pure Python + NumPy port of the THRML (Thermodynamic Hypergraphical Model Library) framework. This implementation removes the JAX dependency entirely, making thermodynamic computing concepts accessible on any platform without GPU drivers, Metal compatibility issues, or heavyweight dependencies.

---

## Why This Exists

The original THRML framework, developed by Extropic for simulating Thermodynamic Sampling Units (TSUs), depends on JAX -- a powerful but platform-sensitive library that introduces GPU driver requirements, Metal compatibility issues on macOS, and a steep setup curve for newcomers.

This port strips the framework down to its mathematical core: NumPy arrays, pure Python control flow, and clean abstractions. The result is a codebase that runs anywhere Python runs, is simple enough to read and modify for learning purposes, and still faithfully implements the Block Gibbs sampling engine at the heart of THRML.

If you want to understand how thermodynamic computing works -- how spin nodes interact through coupling weights, how blocks of variables are sampled in parallel, how Ising models encode optimization problems as energy landscapes -- this implementation lets you do that without fighting your toolchain.

---

## Architecture

```
thrml-python-implementation/
  thrml/
    __init__.py             -- Public API (all core exports)
    nodes.py                -- Node types: SpinNode, CategoricalNode, AbstractNode
    blocks.py               -- Block and BlockSpec (variable grouping for parallel sampling)
    samplers.py             -- Conditional samplers: BernoulliSampler, CategoricalSampler
    block_sampling.py       -- Block Gibbs sampling engine (BlockGibbsSpec, BlockSamplingProgram)
    interaction.py          -- Interaction and InteractionGroup (coupling definitions)
    models.py               -- Pre-built models: IsingModel (E = -Sum J_ij s_i s_j - Sum h_i s_i)
  examples/
    ising_demo.py           -- Working Ising model demonstration
  tests/                    -- Unit test suite
  pyproject.toml            -- Package configuration (Python >= 3.8, NumPy >= 1.20)
```

---

## Core Concepts

| Concept | Implementation | Purpose |
|---------|---------------|---------|
| Spin Nodes | `SpinNode` | Binary variables (+1 / -1) that form the basis of Ising models |
| Categorical Nodes | `CategoricalNode` | Multi-valued variables for problems like Sudoku or graph coloring |
| Blocks | `Block`, `BlockSpec` | Groups of nodes sampled together in parallel (via graph coloring) |
| Samplers | `BernoulliSampler`, `CategoricalSampler` | Conditional probability samplers for each node type |
| Interactions | `Interaction`, `InteractionGroup` | Coupling weights between nodes (the "energy landscape") |
| Block Gibbs Engine | `BlockGibbsSpec`, `BlockSamplingProgram` | The core sampling loop with warmup and thinning |
| Ising Model | `IsingModel` | Pre-built model: arbitrary graph with biases and coupling weights |

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python (>= 3.8) |
| Core Dependency | NumPy (>= 1.20) |
| Dev Dependencies | pytest, matplotlib (optional) |
| GPU Required | No |

---

## Quick Start

```bash
# Install from source
pip install -e .

# Run the Ising model demo
python examples/ising_demo.py

# Run tests
python -m pytest tests/
```

### Example: 10x10 Ising Lattice

```python
import numpy as np
from thrml import SpinNode, IsingModel, sample_states

# Create a 10x10 lattice of spin nodes
n = 10
nodes = [SpinNode() for _ in range(n * n)]

# Define grid edges (right and down neighbors)
edges = []
for i in range(n):
    for j in range(n):
        idx = i * n + j
        if j < n - 1:
            edges.append((idx, idx + 1))
        if i < n - 1:
            edges.append((idx, idx + n))

# Build model with strong ferromagnetic coupling
model = IsingModel(nodes, edges, weights=np.full(len(edges), 2.5), beta=1.0)

# Create sampling program and run
program = model.create_sampling_program()
rng = np.random.default_rng(42)
init_states = [rng.random(len(b)) < 0.5 for b in program.gibbs_spec.free_blocks]
samples = sample_states(rng, program, init_states, [],
                        n_warmup=100, n_samples=1000, thin=10)

# Compute magnetization
for block_samples in samples:
    spins = block_samples.astype(float) * 2 - 1
    mag = np.mean(spins, axis=1)
    print(f"<|M|> = {np.mean(np.abs(mag)):.3f}")
```

---

## Performance

Benchmarked on Apple M1 Max:

| Problem | Time per Sweep |
|---------|---------------|
| 100x100 Ising lattice | ~50ms |
| With block parallelization (graph coloring) | 3-5x logical speedup |

For CPU-only workloads, NumPy performance is competitive with JAX. JAX gains its advantage on GPU, but GPU support (particularly Metal on macOS) introduces the platform dependencies this port was designed to avoid.

---

## Features

- Block Gibbs sampling with warmup and thinning
- Spin nodes (Ising model) and categorical nodes
- Graph coloring for block parallelization
- Ising model on arbitrary graphs with configurable biases and coupling weights
- Clean, documented source suitable for learning

---

## Author

**Huang Akai (Kai)**
Founder @ Universal FAW Labs | Creative Technologist | Ex-Ogilvy | 15+ years experience
