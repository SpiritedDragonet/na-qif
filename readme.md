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
│                                   # - apply_bond_op() : two-site local update
│                                   # - apply_one_site_gate() : single-site unitary
│                                   # - apply_kraus_one_site() : Kraus trajectory
│                                   # - apply_kraus_two_site() : two-site Kraus
│                                   # - swap_sites() : SWAP conveyor belt
│                                   # - get_reduced_density() : uses get_rho_segment
│                                   # - chi property : bond dimensions
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
│   │   └── number_op()           # N = a^† a
│   │
│   └── sites.py                  # Custom Site types (NOT BosonSite)
│       └── TimeBinSite           # 18D site for 780x1517 bin space
│
├── physics/
│   ├── __init__.py
│   ├── gates.py                  # All unitary gate factories
│   │   ├── emission_gate()       # U_emit: atom-photon coupling (54x54)
│   │   ├── qfc_gate()            # U_qfc: 780→1517 conversion (18x18)
│   │   ├── bs_gate()             # U_BS: 50/50 beam splitter (324x324)
│   │   ├── jones_gate()          # U_pol: Jones rotation (18x18)
│   │   ├── jones_gate_from_array() # U_pol from 2x2 Jones matrix
│   │   └── swap_gate()           # W_swap: SWAP gate
│   │
│   └── channels.py               # All Kraus channels
│       ├── loss_channel_1517()           # 1517nm amplitude damping
│       ├── loss_channel_both_subspaces() # Combined 780+1517 loss
│       ├── loss_channel_780_general()    # 780nm loss (internal)
│       ├── detection_channel()           # Single-mode on/off POVM
│       ├── detection_povm_single_site()  # H+V detectors per site (4 outcomes)
│       ├── detection_channel_two_mode()  # Two-port detection (16 outcomes)
│       ├── dephasing_channel()           # Atomic dephasing
│       └── FiberChannelParams            # Fiber drift model (Jones + loss)
│
├── config.py                     # All parameter classes
│   ├── TimeGrid                  # dt, N, t[n]
│   ├── EmitParams                # gamma(t), Alpha matrix
│   ├── QFCParams                 # theta_H, theta_V
│   ├── FiberParams               # Jones matrix, PMD parameters
│   └── DetParams                 # eta_det, p_dark, success_patterns
│
├── simulation/
│   ├── __init__.py
│   ├── trajectory.py             # Single trajectory execution
│   │   ├── TrajectoryRunner       # "Conveyor belt" main loop
│   │   │   ├── initialize_mps()
│   │   │   ├── run_emission()     # SWAP conveyor belt protocol
│   │   │   └── run_bin(n)         # Complete flow for one bin
│   │   │
│   │   ├── EmissionResult         # Result container for emission stage
│   │   │
│   │   └── apply_* functions:     # Unified processor interface
│   │       ├── apply_qfc()        # QFC to all bins
│   │       ├── apply_jones()      # Jones rotation to all bins
│   │       ├── apply_loss()       # Loss channel (1517 only)
│   │       ├── apply_loss_combined() # Loss channel (780+1517)
│   │       ├── apply_fiber_channel() # Jones + loss with random sampling
│   │       ├── apply_bs()         # Beam splitter to all bin pairs
│   │       ├── apply_detection()  # Detection POVM to all bins
│   │       └── find_bsm_success() # Check for BSM heralding patterns
│   │
│   └── simulator.py              # Multi-trial statistics
│       └── run_simulation()       # Returns p_succ ± stderr, F_cond ± stderr
│
├── visualization/
│   ├── __init__.py
│   ├── wavepacket.py             # Wave packet visualization
│   │   └── plot_dual_arm_heatmap() # Dual-arm state heatmap
│   └── state.py                  # State visualization utilities
│
└── tests/                        # Unit tests (correctness checks)
    ├── test_bs_closure.py        # |1>|1> stays in trunc space
    ├── test_kraus_completeness.py # sum(K^† K) = I
    └── ...

outputs/                          # Simulation output directory
└── <YYYYMMDD_HHMM>/              # Timestamped output folders
    ├── 1_after_emission.png
    ├── 2_after_qfc.png
    └── 3_after_fiber.png
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

## "SWAP Conveyor Belt" Protocol

The simulation uses a SWAP conveyor belt approach where atoms move through the chain:

### Chain Layout

**Initial:**
```
atomA, atomB, A1, B1, A2, B2, ..., AN, BN
```

**After SWAP conveyor belt (emission complete):**
```
A1, B1, A2, B2, ..., AN, BN, atomA, atomB
```

This layout allows adjacent (A_n, B_n) pairs for beam splitter and detection operations.

### Simulation Pipeline

```python
# (1) Emission: SWAP conveyor belt protocol
result = run_dual_atom_emission(n_bins=100, ...)
# Result: atoms at end, bins contain 780nm photons

# (2) QFC: 780nm → 1517nm frequency conversion
apply_qfc(mps, n_bins, theta_H=π/4, theta_V=π/4)
# Result: photons converted to telecom wavelength

# (3) Fiber Channel: polarization drift + loss
apply_fiber_channel(mps, n_bins, fiber_params, rng)
# - Jones rotation (random SU(2) sampling)
# - 780nm filtering (100% loss)
# - 1517nm transmission loss (~57%)

# (4) Beam Splitter: interfere A_n with B_n
apply_bs(mps, n_bins)
# Result: HOM interference at each bin pair

# (5) Detection: on/off photon detection
apply_detection(mps, n_bins, eta_det, p_dark, rng)
# Returns: [(dA_H, dA_V, dB_H, dB_V), ...] for each bin

# (6) BSM Heralding: check for success patterns
success, bin_idx, bell_state = find_bsm_success(outcomes)
# Success patterns:
#   Ψ+: (1,0,0,1) or (0,1,1,0)
#   Ψ-: (1,0,1,0) or (0,1,0,1)
```

### FiberChannelParams

Models realistic fiber transmission with random drift:

```python
fiber_params = FiberChannelParams(
    polarization_model="perturb",  # "haar", "perturb", or "euler"
    polarization_sigma=0.1,        # Rotation angle std (rad)
    eta_mean=0.57,                 # Mean transmissivity
    eta_std=0.02,                  # Transmissivity fluctuation
    phase_drift_std=0.2,           # Inter-arm phase drift (rad)
)

# Sample parameters for one trajectory
U_A, U_B, eta, phase = fiber_params.sample_all(rng)
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
