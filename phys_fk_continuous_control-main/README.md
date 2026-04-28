# Pi-ARSQ / Pi-CQN_AS — Physics-Informed Geometric Regularization

This directory contains the reference implementation of the geometric regularizer
proposed in *Physics Informed Learning for Continuous Control Policies*, applied
to two hierarchical value-based backbones:

- **`Pi-ARSQ`** — autoregressive Soft Q-Network (Liu *et al.*, 2025).
- **`Pi-CQN_AS`** — coarse-to-fine Q-Network with Action Sequences
  (Seo *et al.*, 2024).

The regularizer enforces a one-sided HJB viscosity-subsolution constraint
(Eq. 6 in the paper) on the induced state-value function. It is implemented as a
Walk-on-Spheres style local TD residual evaluated on perturbed kinematic states,
and is enabled through a single flag (`use_fk_reg=true`).

## Sub-modules

| Path        | Domain                                            | Backbones                  |
|-------------|---------------------------------------------------|----------------------------|
| [`d4rl/`](d4rl)       | Offline locomotion (Hopper, Walker, HalfCheetah) | `Pi-ARSQ`                  |
| [`rlbench/`](rlbench) | Pixel-based RLBench manipulation (15 tasks)      | `Pi-ARSQ`, `Pi-CQN_AS`     |

Each sub-module is self-contained: a separate conda environment, dependencies,
and entry-point scripts. Refer to the README inside each sub-folder for
benchmark-specific installation and run instructions.

## Geometric Regularizer Hyperparameters

| Flag (Hydra)        | Description                                              | Paper default |
|---------------------|----------------------------------------------------------|--------------:|
| `use_fk_reg`        | Enable the physics-informed regularization              | `true`        |
| `fk_weight`         | λ — coefficient of `L_reg` added to the critic loss     | `1.0`         |
| `kappa`             | Step factor for the Walk-on-Spheres update              | `0.1`         |
| `nu`                | Local cost upper bound `c_local` (ε in the paper)       | `0.01`        |
| `num_walks`         | Number of perturbation samples per state                | `10`          |

These match the configurations used to produce Tables I and II of the paper.

## Reproducing the Paper Tables

- **Table I (RLBench, 15 tasks)** — see [`rlbench/train_ablations.sh`](rlbench/train_ablations.sh)
  and [`rlbench/manager.sh`](rlbench/manager.sh).
- **Table II (D4RL, 5 datasets)** — see [`d4rl/d4rl_manager.sh`](d4rl/d4rl_manager.sh)
  and [`d4rl/d4rl_runner.sh`](d4rl/d4rl_runner.sh).
- **Table III (Episode length, RLBench)** — re-evaluates Table I checkpoints
  using the standard RLBench eval loop in `rlbench/arsq_rlb/runner/`.

The Eikonal baseline (Tables I–II, columns `*_Eikonal`) is obtained with
`use_eikonal=true` and `use_fk_reg=false`; the unregularized backbone is
recovered with both flags set to `false`.

## License

Released under the MIT License (see `LICENSE`). Includes upstream code from the
[ARSQ](https://github.com/CornellRL/ARSQ) and
[CQN-AS](https://github.com/younggyoseo/CQN-AS) repositories.
