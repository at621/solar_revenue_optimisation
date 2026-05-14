# CLAUDE.md

Project-specific guidance for Claude Code sessions on `solarbess` — a synthetic solar+battery
trading simulator with stochastic Pyomo+HiGHS optimisation.

## Background

Read [`README.md`](README.md) first for the architecture map and quick-start. The full design
math is in [`docs/plan.md`](docs/plan.md); the architectural reference is
[`docs/Solar_Battery_Simulator_Plan.pdf`](docs/Solar_Battery_Simulator_Plan.pdf). All six phases
of that plan are implemented. The Estonian price-forecasting PDF in `docs/` is **out of scope**
for this repo (kept for reference if real-data ingestion is added later).

## Environment

- Python **3.11** (pinned in `.python-version`); managed with **uv**.
- Pyomo 6.10 + highspy 1.14 via `pyomo.contrib.appsi.solvers.Highs`.
- Run anything through `uv run python …` or `uv run pytest …`.
- A stray `VIRTUAL_ENV` warning may appear in PowerShell — harmless; uv uses the project `.venv`.

## Commands you'll use most

```bash
uv run pytest -q                                              # 50/50, ~30s
uv run pytest tests/integration/test_phase4_milp.py -v        # focused
uv run python scripts/run_episode.py --policy oracle --seed 7
uv run python scripts/run_evaluation.py --n-episodes 20 --base-seed 100
uv run python scripts/build_demo_notebook.py                  # rebuilds + executes the walkthrough
```

## Code conventions

- **Single source of truth = `configs/default.yaml`**, validated by `solarbess.config.AppConfig`
  (pydantic v2). Don't hardcode magic numbers anywhere else.
- **All timestamps internal = UTC**, tz-aware. Only convert to `Europe/Tallinn` for display or
  for market-time gates (DA gate at 12:00 local).
- **15-minute MTU = 96 periods/day** (configurable). DST transition days raise `DSTTransitionError`
  — MVP does not handle 92/100-MTU days.
- **`m.Omega` not `m.O`** in Pyomo (ruff E741 flags `O` as ambiguous).
- **`pyo.value(...)` only after** `np.isfinite(solve_result.objective)` — `solve_highs` returns
  `nan` on infeasible/exception and `load_solution=False` is set so variable values aren't loaded.
- **No emojis in code or docs** unless explicitly requested.
- **Markdown link format** for file references in user-facing text:
  `[clock.py:42](src/solarbess/clock.py#L42)`.

## The information firewall (load-bearing)

Every policy receives **only** `InfoSet`, `AssetState`, `Position`. The *only* class allowed to
hold a `TruthPath` is `OraclePolicy` (it's the perfect-foresight upper bound). The Phase 5
`PyomoPolicy.attach_mpc_truth(truth)` is a secondary exception — it's a runtime attachment, not
a constructor parameter, and the policy can only read future state through
`ForecastGenerator.generate(now=...)` which only returns scenarios for `t ≥ now`.

If you add a new policy class:
1. Constructor must NOT have a `truth` parameter (enforced by `tests/unit/test_oracle_firewall.py`
   via `inspect.signature`).
2. Callbacks must accept `InfoSet` + `AssetState` + `Position` only.
3. Add it to the firewall test.

## Domain gotchas

- **L2/L3 shared shock**: the solar–DA price coupling is implemented in
  [`src/solarbess/world/layers.py`](src/solarbess/world/layers.py): L3 uses `solar / peak_mw`
  with coefficient `−3.0 · beta_load`. If you change L2's scale, retune this coefficient or the
  `test_midday_solar_da_correlation_negative` test will weaken. Target correlation is
  population −0.30; sample threshold is −0.20.
- **SoC balance ordering**: in [`src/solarbess/optim/helpers.py:add_soc_balance`](src/solarbess/optim/helpers.py),
  precompute `t_index_map = {t: i for i, t in enumerate(t_list)}` once. Never call
  `T_list.index(t)` inside a Pyomo rule.
- **Imbalance sign convention**: `physical − schedule = imb_pos − imb_neg`, both ≥ 0. Objective
  term is `+π_surplus · imb_pos − π_shortfall · imb_neg`. `ProfitAccountant.settle_period` and
  the Pyomo objective must agree to the cent — verified by `tests/unit/test_settlement.py`.
- **DA vs MTU resolution**: DA market is hourly (24 prices); we broadcast hourly DA price to its
  four MTUs at settlement, and (conceptually) submit `q_DA[hour] = mean(q_DA_mtu[4 MTUs])`. See
  `Clock.broadcast_hourly_to_mtu` / `aggregate_mtu_to_hourly_mean`.
- **Probability normalisation**: always re-normalise `probs = probs / probs.sum()` at boundaries;
  assert `abs(sum − 1) < 1e-9` (already done in `ScenarioForecast.__post_init__` and
  `ScenarioTree.__post_init__`).
- **Solar peak time**: in the truth grid, MTU 48 is local noon (= solar peak for our Tallinn
  latitude). Test slicing `[48:60]` covers the 3-hour midday window in local time.

## Phase-specific notes

- **Phase 4 (MILP)**: flip `binary_battery_mode=True` on either builder or on `PyomoPolicy`.
  Scales poorly with scenario count; default-off for stochastic evaluations.
- **Phase 5 (rolling MPC)**: `PyomoPolicy(mpc_dispatch_horizon=-1, mpc_dispatch_period=K,
  forecast_gen=fg, forecast_seed=…)`. `H = -1` means "to end of day"; finite H needs terminal
  SoC value penalties or the battery will deplete greedily. The episode loop calls
  `policy.attach_mpc_truth(truth)` if the method exists.
- **Phase 6 (scenario tree)**: `TreePolicy(n_ida_nodes=K)`. Tree built by sort-and-split on
  `stage1_feature` (default: mean DA price in first half of day). Variables are
  `q_da[t]` (root), `q_id_ida[ida, t]` (per IDA branch), leaf-level dispatch.
  Non-anticipativity is enforced by construction; verified by
  `tests/integration/test_phase6_tree.py::test_tree_enforces_non_anticipativity`.

## When making changes

- **Don't break the firewall test** — if a new feature seems to want the truth, channel it
  through `InformationOracle.view(...)` or `ForecastGenerator.generate(now=...)`.
- **Re-run pytest after every non-trivial change**. The full suite is ~30s.
- **Re-run the notebook after API changes** in: `policy/*`, `optim/*`, `forecast/*`,
  `simulator/episode.py`, `config.py`. The build script `scripts/build_demo_notebook.py`
  rebuilds and executes; if any cell errors, the script fails loudly.
- **Don't add emojis, comments stating the obvious, or speculative abstractions**. The codebase
  is intentionally compact (~3000 LOC across all phases). When in doubt, write the smallest
  change that passes tests.

## Things that are NOT yet implemented

- Real-data ingestion (Nord Pool, Elering, ENTSO-E, ECMWF). Sketched in the Estonian
  forecasting PDF; would replace the synthetic `TruthGenerator` with a vintage-safe data store.
- Continuous intraday market (current model: 3 fixed gates at t-6h/t-3h/t-1h before delivery
  starts; all rounds happen pre-delivery).
- Bayesian conditional forecasts using full Gaussian conditioning on revealed past observations.
  The infrastructure is there (`forecast/error_model.py` already has the correlation matrices);
  see Section 12 of [`notebooks/walkthrough.ipynb`](notebooks/walkthrough.ipynb) for context.
- DST transition days (92/100-MTU schedules).
- Scenario reduction for the tree (current sort-and-split keeps all base scenarios as leaves).

## Where to look first

- Mathematical formulation: [`docs/plan.md`](docs/plan.md) §8 (objective) and §9 (CVaR).
- Pyomo model: [`src/solarbess/optim/builder.py`](src/solarbess/optim/builder.py) and
  [`src/solarbess/optim/tree_builder.py`](src/solarbess/optim/tree_builder.py).
- Episode loop: [`src/solarbess/simulator/episode.py`](src/solarbess/simulator/episode.py).
- A complete end-to-end example with plots: [`notebooks/walkthrough.ipynb`](notebooks/walkthrough.ipynb).
