# Pedagogical Sovereignty — MORL Proof-of-Concept

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Dependencies](https://img.shields.io/badge/Dependencies-numpy%20%7C%20matplotlib%20%7C%20scipy-orange)]()

> **Companion simulation code for:**
>
> *Pedagogical Sovereignty and Its Contradictions: Toward a Postdigital Critique of Decolonial AI in Education via Multi-Objective Reinforcement Learning*
>


## ⚠️ Ethical Disclaimer — Read Before Using

This is a **synthetic simulation only**.

- No real student data was collected at any stage
- No workshops were conducted with any community
- No Anishinaabe participants were involved in any capacity
- All student states, interactions, and learning outcomes are **computationally generated**
- Anishinaabe pedagogical values used in this simulation are informed **solely by published scholarly literature** cited in the paper

The Anishinaabe geometry tutor described here is a **conceptual thought experiment**, not a deployed or tested system. Any real implementation would require extensive relationship-building, free and prior informed consent from relevant community governance structures, fair compensation of community partners, and ongoing accountability to those communities throughout and beyond any project lifecycle — consistent with Smith (2012), *Decolonizing Methodologies*.

---

## Overview

This repository contains the proof-of-concept implementation demonstrating how **Multi-Objective Reinforcement Learning (MORL)** can create a technically pluralistic substrate for educational AI — one that makes pedagogical value trade-offs explicit and navigable rather than hiding them inside a single optimization metric.

The simulation operationalizes four Anishinaabe-informed pedagogical values as a vector reward function:

| Dimension | Symbol | Pedagogical Meaning |
|-----------|--------|---------------------|
| Relational | `r_rel` | Understanding geometric forms through relationships to natural and community context |
| Holistic | `r_hol` | Connecting concepts to cultural context, stories, and land-based practices |
| Observational | `r_obs` | Deriving principles from concrete examples before abstract rules |
| Autonomy | `r_aut` | Learner-directed, non-linear inquiry |

The core architectural claim is that **single-objective RL** hides its value choices behind a single metric, systematically disadvantaging non-mainstream learners, while **MORL** makes those choices explicit, auditable, and politically contestable.

---

## What the Code Produces

Running the simulation generates **6 publication-quality figures** that directly support the paper's claims:

| Figure | File | Maps to Paper |
|--------|------|---------------|
| Pareto Frontier | `fig1_pareto_frontier.png` | §3.2, Figure 2 |
| Reward Radar | `fig2_reward_radar.png` | §3.2, Table 1 |
| Steerable Alignment | `fig3_steerable_alignment.png` | §4.3, Table 2 |
| Training Curves | `fig4_training_curves.png` | §3.2 |
| Multi-Seed Comparison | `fig5_equity_comparison.png` | §5.1 |
| Per-Group Equity | `fig6_group_equity.png` | §5.1 |

Plus `results_summary.txt` with all numeric results, statistical tests, and the action dictionary.

---

## Requirements

```
Python 3.8+
numpy
matplotlib
scipy
```

Install dependencies:

```bash
pip install numpy matplotlib scipy
```

No GPU required. No deep learning frameworks required. Runs on any standard laptop.

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/pedagogical-sovereignty-morl.git
cd pedagogical-sovereignty-morl

# Install dependencies
pip install numpy matplotlib scipy

# Quick test run (~4 minutes)
python ps_morl.py --fast

# Full publication-quality run (~35 minutes)
python ps_morl.py --seeds 10 --weights 80 --episodes 3500
```

All outputs are saved to `./ps_figs/` by default.

---

## Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--outdir` | `./ps_figs` | Output directory for figures and results |
| `--seeds` | `6` | Number of random seeds for statistical comparison |
| `--episodes` | `3000` | Training episodes per agent |
| `--weights` | `60` | Weight vectors for Pareto frontier sweep |
| `--ep_len` | `22` | Episode length (tutoring interactions per session) |
| `--decay` | `0.10` | Mastery decay probability per step |
| `--fast` | `False` | Quick test mode (fewer episodes and weights) |

### Example runs

```bash
# Minimal test — confirms everything works (~2 min)
python ps_morl.py --fast --seeds 3 --outdir ./test_output

# Balanced quality/speed (~15 min)
python ps_morl.py --seeds 6 --weights 40 --episodes 2000

# Publication quality — use for final figures (~35 min)
python ps_morl.py --seeds 10 --weights 100 --episodes 3500 --outdir ./final_figs
```

---

## Repository Structure

```
pedagogical-sovereignty-morl/
│
├── ps_morl.py              # Main simulation — all code in one file
├── README.md               # This file
├── LICENSE                 # MIT License
│
└── ps_figs/                # Generated outputs (created on first run)
    ├── fig1_pareto_frontier.png
    ├── fig2_reward_radar.png
    ├── fig3_steerable_alignment.png
    ├── fig4_training_curves.png
    ├── fig5_equity_comparison.png
    ├── fig6_group_equity.png
    └── results_summary.txt
```

---

## Simulation Design

### Environment

The tutoring environment is a finite-horizon MDP:

- **State space**: `mastery ∈ {0…6} × engagement ∈ {0…4}` → 35 discrete states
- **Action space**: 8 pedagogical actions (see table below)
- **Mastery decay**: `p = 0.10` per step — creates genuine equilibrium dynamics and ensures the Pareto frontier is non-trivial
- **Episode length**: 22 interactions

### Why Mastery Decay Matters

Without decay, all agents trivially converge to maximum mastery regardless of policy, producing a flat Pareto frontier with no meaningful trade-off. Decay creates an equilibrium mastery level that depends on the action's mastery gain rate versus decay rate — making policy choice genuinely consequential and producing the downward-sloping Pareto curve the paper describes.

### Actions

| ID | Name | E[Mastery] | EPV Profile |
|----|------|-----------|-------------|
| A0 | state_rule_directly | **0.526** | [0.05, 0.05, 0.08, 0.08] |
| A1 | show_natural_example | 0.495 | [0.80, 0.35, 0.78, 0.28] |
| A5 | show_multiple_examples | 0.472 | [0.42, 0.22, 0.92, 0.32] |
| A6 | guided_observation | 0.475 | [0.52, 0.38, 0.85, 0.62] |
| A4 | offer_learner_choice | 0.414 | [0.32, 0.28, 0.22, 0.92] |
| A2 | connect_to_beadwork | 0.351 | [0.55, 0.88, 0.38, 0.42] |
| A3 | connect_to_land | 0.348 | [0.78, 0.82, 0.48, 0.42] |
| A7 | present_elder_story | 0.341 | [0.82, 0.92, 0.52, 0.48] |

**Key design property**: A0 has the highest expected mastery signal (0.526) because it works well for mainstream learners who make up 55% of the simulated population. This causes the single-objective baseline to preferentially learn A0, creating systematic inequity for observational and relational learners — the core inequity the paper argues MORL mitigates.

### Student Groups

| Group | Population | Responds best to |
|-------|-----------|-----------------|
| Mainstream learners | 55% | Direct instruction (A0) |
| Observational learners | 28% | Example-first actions (A5, A6) |
| Relational learners | 17% | Cultural connection actions (A2, A3, A7) |

### MORL Agent

The agent uses **linear scalarization Q-learning**, maintaining one Q-table per reward dimension. Sweeping weight vectors uniformly over the 4-simplex approximates the **Convex Coverage Set** (Roijers et al., 2013) — the standard approach for MORL under linear utility functions.

This is technically honest: the paper describes a Pareto frontier, and linear scalarization over the simplex is the correct method for approximating that frontier under linear utility assumptions. A more sophisticated implementation using Pareto-Q-learning or envelope Q-learning would produce a more complete frontier but would not change the conceptual argument.

### Equity Metric

Equity is measured as `1 − Gini coefficient` across group mean rewards. A Gini of 0 means perfect equity across groups; a Gini of 1 means maximal inequality. The simulation consistently shows:

- **Baseline (mastery-only)**: higher Gini, lower equity — because A0 dominates and A0 systematically disadvantages non-mainstream learners
- **MORL (balanced weights)**: lower Gini, higher equity — because cultural actions score higher on EPV dimensions, diversifying the policy

---

## Interpreting the Results

### What the results do support

- MORL's architecture produces a **genuine, navigable Pareto trade-off** between mastery and equity in this simulated environment
- A **single-objective baseline systematically disadvantages** non-mainstream learners when the majority group responds better to the action with the highest average mastery signal
- **Different weight configurations produce qualitatively different pedagogical behaviors** from the same underlying model — demonstrating steerable alignment
- The **equity improvement** of MORL over the baseline is consistent across seeds (Cohen's d ≈ 0.985 in a representative run)

### What the results do NOT support

- That this equity improvement would transfer to real educational contexts with real students
- That the four reward dimensions accurately capture what Anishinaabe community members would actually value in a geometry tutor
- That the simulated student groups accurately represent the diversity of real Anishinaabe learners
- That optimizing for these reward signals would not produce unexpected effects on community life and knowledge transmission

These limitations are not failures of the simulation — they are the honest limits of any synthetic proof-of-concept. The simulation demonstrates **architectural feasibility**, not educational validity.

---

## Citing This Work

If you use this code, please cite the companion paper:

```
[Author(s)] (under review). Pedagogical Sovereignty and Its Contradictions:
Toward a Postdigital Critique of Decolonial AI in Education via
Multi-Objective Reinforcement Learning.
Postdigital Science and Education. Major Revision.
```

And the foundational MORL reference for the Convex Coverage Set approach:

```
Roijers, D. M., Vamplew, P., Whiteson, S., and Dazeley, R. (2013).
A survey of multi-objective sequential decision-making.
Journal of Artificial Intelligence Research, 48, 67–113.
```

---

## License

MIT License. See `LICENSE` for details.

Note: The pedagogical values, reward profiles, and action descriptions in this simulation are informed by published scholarly literature on Anishinaabe pedagogy. They are not a product of research with Anishinaabe communities, do not represent any community's endorsed position, and should not be treated as authoritative descriptions of Anishinaabe educational traditions. Any use of this code in educational contexts involving Indigenous communities must involve genuine community partnership from the outset.

---

## Contributing

This repository is a companion to a specific paper under review. Issues and pull requests are welcome for:

- Bug fixes
- Improved MORL algorithms (e.g., Pareto-Q-learning, envelope Q-learning)
- Additional equity metrics
- Visualization improvements

Please do not submit pull requests that modify the core reward profiles or student group parameters without discussion, as these are tied to the paper's theoretical claims.

---

