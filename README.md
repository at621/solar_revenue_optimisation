# solarbess

A closed-loop synthetic simulator for **solar + battery storage** day-ahead and intraday trading
under price and solar uncertainty. Built end-to-end against the design in
[`docs/Solar_Battery_Simulator_Plan.pdf`](docs/Solar_Battery_Simulator_Plan.pdf) and the math in
[`docs/plan.md`](docs/plan.md). World statistics calibrated to Estonia post-Feb 2025; no real
market data is ingested (synthetic-only).

All six phases of the original plan are implemented:

| Phase | What it adds | Tests |
|---|---|---|
| 1 — Deterministic | Pyomo+HiGHS LP on a single forecast path | 37 |
| 2 — Stochastic | Extensive-form scenario LP, paired-seed eval, paired Wilcoxon | (in 37) |
| 3 — Risk | CVaR + imbalance penalty + battery degradation | (in 37) |
| 4 — MILP | `binary_battery_mode` forbids simultaneous charge/discharge | 4 |
| 5 — Rolling MPC | Re-solve a shrinking-tail problem at every dispatch step with a fresh forecast | 4 |
| 6 — Scenario tree | Multi-stage non-anticipativity (per-IDA-branch q_id, not per-leaf) | 5 |
| **Total** | | **50** |

## Architecture (eight modules)

```
┌──────────────────────────┐         ┌──────────────────────────┐
│ TruthGenerator (Module 1)│         │ Forecast generator       │
│  5-layer synthetic world │         │  truth + structured error│
└────────────┬─────────────┘         └────────────┬─────────────┘
             │ sealed TruthPath                   │
             ▼                                    ▼
        ┌────────────────────────────────────────────┐
        │ Information Oracle (Module 2)              │
        │  the ONLY legal channel world → policy     │
        │  view(decision_time, now) -> InfoSet       │
        └──────────────────┬─────────────────────────┘
                           │ InfoSet
                           ▼
   ┌───────────────────────────────────────────────────────┐
   │ Policy (Module 5) - 3 callbacks                       │
   │   decide_day_ahead  / decide_intraday / decide_dispatch│
   │   { RuleBasedPolicy, PyomoPolicy, TreePolicy, Oracle }│
   └────────┬──────────────────────────────────────────────┘
            │                                       ▲
            ▼                                       │
   ┌────────────────────┐    ┌──────────────────────┴────────┐
   │ MarketInterface (4)│    │ AssetModel (Module 3)         │
   │ Position + clearing│    │ Battery SoC, solar-only charge│
   └────────┬───────────┘    └──────────┬────────────────────┘
            │                           │
            ▼                           ▼
   ┌────────────────────────────────────────────┐
   │ ProfitAccountant (Module 6)                │
   │  per-period PnL decomposition              │
   └────────┬──────────────────────────┬────────┘
            │                          │
            ▼                          ▼
   ┌────────────────────┐    ┌──────────────────────┐
   │ Reporting (7)      │    │ Eval harness (8)     │
   │ HTML, plots        │    │ paired seeds, tests  │
   └────────────────────┘    └──────────────────────┘
```

## Quick start

```bash
# 1. Install (uv + Python 3.11)
uv sync --extra dev --extra notebook

# 2. Run one episode with a chosen policy
uv run python scripts/run_episode.py --policy pyomo_stochastic --seed 42

# 3. Run a paired N-episode evaluation
uv run python scripts/run_evaluation.py --n-episodes 20 --base-seed 100

# 4. World validation (200 days marginals + corr panel)
uv run python scripts/validate_world.py

# 5. All policies side-by-side on one day
uv run python scripts/plot_episode.py --date 2026-06-15 --seed 42

# 6. Full test suite
uv run pytest -q

# 7. Walkthrough notebook (40 cells, all six phases)
uv run python scripts/build_demo_notebook.py
# then open notebooks/walkthrough.ipynb
```

## Project layout

```
optimise_pricing_claude/
├── README.md                  this file
├── CLAUDE.md                  project guide for Claude Code sessions
├── pyproject.toml             uv project, deps, pytest, ruff
├── .python-version            3.11
├── configs/
│   └── default.yaml           single source of truth for parameters
├── docs/
│   ├── plan.md                detailed math + Pyomo formulation
│   ├── Solar_Battery_Simulator_Plan.pdf   eight-module architecture deck
│   └── estonian_da_price_forecasting.pdf  out-of-scope research track
├── notebooks/
│   └── walkthrough.ipynb      executable tour of all six phases
├── scripts/
│   ├── run_episode.py         single episode + HTML report
│   ├── run_evaluation.py      paired N-episode harness + comparison report
│   ├── validate_world.py      marginals + L2/L3 correlation
│   ├── plot_episode.py        all policies on one day
│   └── build_demo_notebook.py rebuilds + executes walkthrough.ipynb
├── src/solarbess/
│   ├── config.py              Pydantic v2 models + YAML loader
│   ├── clock.py               MTU grid, DA gate, DST guard, hour↔MTU
│   ├── rng.py                 paired-seed plumbing
│   ├── world/
│   │   ├── layers.py          L1–L5 sampling functions
│   │   ├── truth.py           TruthGenerator → TruthPath
│   │   └── oracle.py          InformationOracle (firewall)
│   ├── forecast/
│   │   ├── error_model.py     lead-time variance + Matern correlation
│   │   ├── scenarios.py       ForecastGenerator + ScenarioForecast
│   │   └── tree.py            Phase 6: scenario tree clustering
│   ├── asset/
│   │   ├── battery.py         AssetModel.step + solar-only guard
│   │   └── pv.py              inverter clip
│   ├── market/
│   │   ├── interface.py       Position + MarketInterface
│   │   └── settlement.py      ProfitAccountant
│   ├── policy/
│   │   ├── base.py            Policy Protocol (3 callbacks)
│   │   ├── rule_based.py
│   │   ├── pyomo_policy.py    Phase 1-3 + 4 (MILP) + 5 (rolling MPC)
│   │   ├── oracle_policy.py   perfect-foresight upper bound
│   │   └── tree_policy.py     Phase 6: multi-stage tree
│   ├── optim/
│   │   ├── helpers.py         SoC, imbalance, no-simultaneous helpers
│   │   ├── builder.py         deterministic + stochastic builders
│   │   ├── tree_builder.py    Phase 6: node-indexed tree model
│   │   ├── cvar.py            Phase 3: CVaR layer
│   │   └── solve.py           HiGHS APPSI wrapper (robust to infeasible)
│   ├── simulator/
│   │   └── episode.py         daily clock loop
│   ├── eval/
│   │   ├── harness.py         paired-seed N-episode runner
│   │   └── stats.py           paired t, Wilcoxon, empirical CVaR
│   └── reporting/
│       ├── episode_report.py
│       ├── comparison_report.py
│       └── world_validation.py
└── tests/
    ├── unit/                  9 files
    ├── integration/           5 files (incl. phase4, phase5, phase6)
    └── property/              1 file
```

## Key design decisions

- **Information firewall.** Policies receive only `InfoSet`; only `OraclePolicy` accepts the
  sealed `TruthPath`. Enforced by `tests/unit/test_oracle_firewall.py` via `inspect.signature`.
- **Same-seed paired runs.** The harness uses identical world+forecast seeds across policies in
  each episode index — variance reduction makes paired Wilcoxon tight at small N.
- **L2/L3 coupling.** Solar and DA price share the residual-load channel in the truth model.
  Sunny days really do depress prices; verified by a 200-day correlation check.
- **SoC ordering.** Pyomo soc-balance uses a precomputed `t_index_map` (O(1) lookup) instead of
  `T_list.index(t)` inside the rule (would be O(T²) build).
- **Solar-only charging.** Default config (`import_limit_mw = 0`) makes this physical; the Pyomo
  builders also add an explicit `charge ≤ solar − curtail` constraint unless overridden.
- **MILP only when needed.** With `degradation_cost > 0` the LP relaxation rarely opens both
  legs; the Phase 4 binary mode is a structural guarantee at MILP solve cost.
- **Tree clustering.** Sort-and-split on a 1-D stage-1 feature (default: mean DA price in the
  first half of the day) — deterministic, no sklearn dep, balanced buckets.

## Verification gates

- `uv run pytest -q` → 50/50 pass (~33s on a workstation).
- World validation: midday solar–DA correlation < −0.20 over 150 samples (PDF target was −0.30).
- Per-day ordering: `oracle ≥ stochastic ≥ deterministic ≥ rule-based` on mean PnL across paired
  episodes; statistically significant at p < 0.01 for oracle vs everything once N ≥ 10.
- Phase 4: no period has both `charge > 0` AND `discharge > 0` under MILP.
- Phase 6: leaves in the same IDA branch share `q_id` at every t.

## Extending

- **Real-data ingestion** (deferred): see [`docs/estonian_da_price_forecasting.pdf`](docs/estonian_da_price_forecasting.pdf)
  for a vintage-safe Nord Pool / Elering / ENTSO-E / ECMWF design.
- **Continuous intraday** market with per-period gate times instead of three fixed rounds.
- **Deeper scenario trees** with multiple branching stages (current tree is two stages).
- **Conditional Bayesian forecasts** at dispatch (current MPC re-samples but does not condition
  on revealed past — straightforward Gaussian conditioning given the existing correlation matrix).
