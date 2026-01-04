# Quantum Simulation: Neutral Atom Quantum Interface

Time-bin MPS simulation for neutral atom quantum entanglement protocol.

## Project Structure

```
atom_sim/
├── __init__.py
│
├── core/
│   ├── __init__.py
│   └── mps.py                    # MPSState container (TeNPy backend)
│                                   # - Tensor network storage
│                                   # - apply_bond_op() : two-site local update
│                                   # - apply_kraus_one_site() : Kraus trajectory
│                                   # - apply_kraus_two_site() : two-site Kraus
│                                   # - finalize_bin_pair() : freeze measured bins
│                                   # - get_reduced_density() : uses get_rho_segment
│
├── hilbert/
│   ├── __init__.py
│   ├── basis.py                  # Space definitions and tensor products
│   │   ├── SubSpace              # Single subspace (780, 1517, atom)
│   │   ├── ProductSpace          # Tensor product space [s1, s2, ...]
│   │   └── subspace_gate()       # Embed subspace gate into product space
│   │
│   ├── operators.py              # Basic operator factory
│   │   ├── annihilation_op()     # a[i] on specified space
│   │   ├── creation_op()         # a^†[i]
│   │   ├── atom_transition()     # S_+, S_- (atomic transition operators)
│   │   └── number_op()           # N = a^† a (for wave packet extraction)
│   │
│   └── sites.py                  # Custom Site types (NOT BosonSite)
│       └── FiniteDimSite         # Generic d-dimensional site
│
├── physics/
│   ├── __init__.py
│   ├── gates.py                  # All unitary gate factories (with caching)
│   │   ├── emission_gate()       # U_emit: exp(√dt * L ⊗ b^† + h.c.) [per bin]
│   │   ├── qfc_gate()            # U_qfc: exp(-iθ * b c^† + h.c.) [cached]
│   │   ├── bs_gate()             # U_BS: 50/50 beam splitter [cached]
│   │   ├── jones_gate()          # U_pol: Jones rotation [cached]
│   │   └── swap_gate()           # W_swap: SWAP gate [cached]
│   │
│   └── channels.py               # All Kraus channels
│       ├── loss_channel()        # Amplitude damping (fiber loss)
│       ├── detection_channel()   # on/off POVM
│       └── dephasing_channel()   # Atomic dephasing
│
├── config.py                     # All parameter classes
│   ├── TimeGrid                  # dt, N, t[n]
│   ├── EmitParams                # gamma(t), Alpha matrix
│   ├── QFCParams                 # theta_H, theta_V
│   ├── FiberParams               # Jones matrix, PMD parameters
│   └── DetParams                 # eta_det, p_dark, success_patterns
│                               #   + pattern_to_bell, pattern_to_correction
│
├── simulation/
│   ├── __init__.py
│   ├── trajectory.py             # Single trajectory execution
│   │   └── TrajectoryRunner       # "Conveyor belt" main loop
│   │       ├── initialize_mps()
│   │       ├── run_bin(n)         # Complete flow for one bin
│   │       └── extract_wave_packet() # Get p_n and xi_n from MPS
│   │
│   └── simulator.py              # Multi-trial statistics
│       └── run_simulation()       # Returns p_succ ± stderr, F_cond ± stderr
│
└── tests/                        # Unit tests (correctness checks)
    ├── test_bs_closure.py        # |1>|1> stays in trunc space
    ├── test_kraus_completeness.py # sum(K^† K) = I
    ├── test_no_noise_limit.py    # F -> 1 when ideal
    ├── test_hom_mismatch.py      # p_succ decreases with mismatch
    └── test_trajectory_consistency.py # statistical convergence
```

## Layer Responsibilities

| Layer | Responsibility | Does NOT care about |
|-------|----------------|---------------------|
| `core/mps.py` | Tensor network storage, local TEBD updates, SVD | Physical meaning |
| `hilbert/` | Linear algebra: spaces, bases, operators | What gates do |
| `physics/` | Physics: gate matrices, Kraus channels | MPS updates |
| `config.py` | Data: parameter storage | Computation |
| `simulation/` | Orchestration: call order, conditions | How matrices are computed |
| `tests/` | Correctness validation | - |

## Data Flow

```
config.py → physics/gates.py → hilbert/basis.py → numpy matrices
                                               ↓
simulation/trajectory.py → core/mps.py → tensor network updates (local only!)
```

## Critical Design Principles

### 1. No `apply_local_op` for non-unitary operations
All Kraus and measurement operations MUST use local theta + SVD updates to avoid canonical sweep. Use:
- `apply_bond_op(i, op)` for two-site gates
- `apply_kraus_one_site(i, {Kμ}, rng)` for single-site Kraus
- `apply_kraus_two_site(i, {Kμ}, rng)` for two-site measurement

### 2. Bin discard mechanism
After measurement, bins are frozen (bond=1) and never accessed again. Use `finalize_bin_pair(n)` to ensure linear complexity.

### 3. Site type: FiniteDimSite, NOT BosonSite
Custom `FiniteDimSite(d)` with operator dictionary, not TeNPy's `BosonSite` which has incompatible semantics.

### 4. Density matrix extraction
Always use `get_rho_segment([i])` (TeNPy's method), never direct contraction of `_B[i]`.

## Physical Model

### Atomic Levels (3D)
- `|e>`: 5P_{3/2}, F'=0, m_F=0 (excited)
- `|0>`: 5S_{1/2}, F=1, m_F=+1 (ground)
- `|1>`: 5S_{1/2}, F=1, m_F=-1 (ground)

### Selection Rules
- `|e> → |0>`: Δm = +1 → σ+ photon (right circular)
- `|e> → |1>`: Δm = -1 → σ- photon (left circular)

### Photon Subspaces
- **780nm**: 3D `{|vac>, |H>, |V>}` (single-photon truncation)
- **1517nm**: 6D `{|vac>, |H>, |V>, |2H>, |2V>, |HV>}` (two-photon truncation)

### Hilbert Space Decomposition

```
System site: H_S = H_atom_A(3D) ⊗ H_atom_B(3D) = 9D
Bin site:    H_bin = H_780(3D) ⊗ H_1517(6D) = 18D
```

**IMPORTANT**: Atoms are in system site ONLY, NOT in bin sites.

### MPS Chain Layout

```
A0 - B0 - A1 - B1 - A2 - B2 - ... - AN - BN
│    │
│    └─ Atom B (3D)
└─ Atom A (3D)

A_n, B_n - Time-bin field sites (arm A and B, bin n), each 18D
```

Alternative (with single system site):
```
S(9D) - A1(18D) - B1(18D) - A2(18D) - B2(18D) - ...
```

## "Time-Bin Carriage" Protocol (per bin n)

Each time-bin is like a carriage passing through gates:

```python
for n in 1..N:
    # (1) Emission: two-site unitary (atom, bin)
    apply_U_2(A0, An, emission_gate_A(n))
    apply_U_2(B0, Bn, emission_gate_B(n))

    # (2) QFC: one-site unitary (780 ↔ 1517 within bin)
    apply_U_1(An, qfc_gate())      # cached
    apply_U_1(Bn, qfc_gate())      # cached

    # (3) Jones rotation: one-site unitary (telecom subspace)
    apply_U_1(An, jones_gate_A())  # cached
    apply_U_1(Bn, jones_gate_B())  # cached

    # (4) Loss: one-site Kraus
    apply_K_1(An, loss_kraus_A(), rng)
    apply_K_1(Bn, loss_kraus_B(), rng)

    # (5) Beam splitter: two-site unitary (An, Bn)
    apply_U_2(An, Bn, bs_gate())   # cached

    # (6) Detection: two-site measurement Kraus
    outcome = apply_M_2(An, Bn, detection_kraus(), rng)
    record(outcome, n)

    # (7) Discard measured bins
    finalize_bin_pair(An, Bn)

    if outcome in success_patterns:
        return SUCCESS, extract_atomic_state(), outcome
```

## Wave Packet Extraction

Wave packet shape is encoded in time-bin occupation probabilities:

```python
# Extract intensity envelope
p_n = <N_H_n + N_V_n> for each bin n

# Extract complex amplitude (for HOM visibility)
xi_n^H = <1_H_n|psi>, xi_n^V = <1_V_n|psi>

# Mode overlap (determines HOM visibility)
M = |sum_n (xi_A_n^H* xi_B_n^H + xi_A_n^V* xi_B_n^V)|^2
```

## Dependencies

- `numpy` - Array operations
- `physics-tenpy` - Tensor network backend

## References

See `docs/` directory for detailed specifications:
- `总设计图纸.md` - Overall architecture
- `逐行流水表.md` - Detailed execution flow
- `要模拟的对象与输出.md` - Implementation specification
- `有关空间排列与直积构造相关修改建议.md` - Design corrections and clarifications
