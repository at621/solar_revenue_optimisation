Yes — that should be a **core design requirement**, not an add-on. I would model the simulator as a **probabilistic stochastic optimization problem**, where every price and solar forecast is represented as either:

1. **prediction intervals / quantiles**, such as P05, P10, P50, P90, P95; or
2. **explicit scenarios with probabilities**, which is the preferred form for the optimizer.

Since you want the exercise implemented in **Pyomo + HiGHS**, I would convert all uncertainty into a **finite scenario ensemble** and solve a stochastic linear or mixed-integer linear optimization model. HiGHS is suitable for this because it supports large-scale sparse LP, MIP, and QP models, and Pyomo has a HiGHS interface through its APPSI solver layer. ([HiGHS][1])

## Revised modelling architecture

The simulator should have four distinct layers:

```text
1. True world simulator
   Generates the hidden true solar production and market prices.

2. Probabilistic forecast generator
   Produces forecast distributions, not point forecasts.

3. Pyomo/HiGHS stochastic optimizer
   Chooses day-ahead and intraday commitments under uncertainty.

4. Market settlement simulator
   Applies realized prices, realized solar production, battery dispatch,
   imbalance settlement, and degradation costs.
```

The optimizer must **never see the true future path**. It should only see scenario forecasts available at the simulated decision time.

---

# 1. Forecast output format

The forecast module should output one of these two formats.

## Option A: quantile forecast output

This is useful for model diagnostics and visualization.

```text
time_interval
solar_P05
solar_P10
solar_P50
solar_P90
solar_P95
DA_price_P05
DA_price_P10
DA_price_P50
DA_price_P90
DA_price_P95
ID_price_P05
ID_price_P10
ID_price_P50
ID_price_P90
ID_price_P95
```

Example:

```text
t                  solar_P10   solar_P50   solar_P90   DA_P10   DA_P50   DA_P90
2026-06-01 10:00   18.2        25.4        31.6        42.0     57.5     91.0
2026-06-01 10:15   19.1        26.2        32.4        40.5     55.0     88.5
```

This is interpretable, but not ideal for optimization because the optimizer also needs **cross-time dependence** and **price-solar dependence**.

## Option B: scenario forecast output

This should be the main input to Pyomo.

```text
scenario_id
probability
time_interval
solar_mw
DA_price_EUR_MWh
ID_price_EUR_MWh
imbalance_surplus_price_EUR_MWh
imbalance_shortfall_price_EUR_MWh
```

Example:

```text
scenario_id  probability  t                  solar_mw  DA_price  ID_price  imb_surplus  imb_shortfall
s001         0.025        2026-06-01 10:00   30.1      48.0      46.5      35.0         72.0
s001         0.025        2026-06-01 10:15   31.3      49.5      48.0      36.0         74.0
s002         0.025        2026-06-01 10:00   15.8      85.0      91.0      70.0         120.0
s002         0.025        2026-06-01 10:15   16.4      88.0      94.0      71.0         125.0
```

This scenario format is what I would use directly in the Pyomo model.

---

# 2. Forecast uncertainty should be path-based

Do **not** generate uncertainty independently for every 15-minute interval. That would make the optimization unrealistic.

Each scenario should be a coherent path:

```text
scenario ω =
    solar path over the day
    day-ahead price path
    intraday price path
    imbalance price path
```

The scenarios should preserve:

```text
solar autocorrelation
cloud-front ramps
price autocorrelation
price spikes
negative prices
price-solar correlation
forecast uncertainty shrinking closer to delivery
seasonality
different uncertainty by hour of day
```

For example, if a scenario has unexpectedly high solar production across Estonia or nearby bidding zones, prices may also be lower. If solar and price uncertainty are simulated independently, the battery optimizer may look artificially good.

---

# 3. Convert error margins into scenarios

If the forecasting model naturally produces error margins such as P10/P50/P90, I would still convert them into scenarios before optimization.

A practical process:

```text
1. Generate marginal quantile forecasts for solar and prices.
2. Estimate historical forecast-error correlations.
3. Sample correlated latent variables using a Gaussian or t-copula.
4. Transform samples through the forecast quantile functions.
5. Produce N scenarios with probabilities.
6. Reduce scenarios if needed for computational speed.
```

Recommended starting sizes:

```text
MVP model:                 20 to 50 scenarios
Research-grade DA model:   100 to 500 scenarios
Rolling intraday model:    20 to 100 scenarios
Fast production model:     scenario reduction to 20 to 40 representative paths
```

The key point is that the optimizer should receive this:

```python
scenarios[omega, t, variable]
probability[omega]
```

not just this:

```python
expected_price[t]
expected_solar[t]
```

---

# 4. Stochastic optimization structure

I would model the decision problem as a **rolling stochastic optimization**.

At each market decision time, the simulator does this:

```text
1. Generate posterior scenarios conditional on information available now.
2. Build or update a Pyomo model.
3. Solve with HiGHS.
4. Execute only the decision that is actually available now.
5. Advance simulation time.
6. Reveal new information.
7. Re-solve.
```

This is important because intraday decisions should benefit from newer, sharper forecasts.

---

# 5. Day-ahead decision problem

At the day-ahead stage, the optimizer chooses:

```text
q_DA[t] = quantity committed in the day-ahead market
```

This is a **here-and-now decision**. It must be the same across all scenarios, because the trader does not yet know which scenario will happen.

Then the model includes hypothetical recourse decisions:

```text
q_ID[ω,t]           intraday correction under scenario ω
charge[ω,t]         battery charging under scenario ω
discharge[ω,t]      battery discharging under scenario ω
soc[ω,t]            battery state of charge under scenario ω
imbalance_pos[ω,t]  surplus imbalance
imbalance_neg[ω,t]  shortfall imbalance
curtail[ω,t]        curtailed solar
```

The day-ahead optimizer maximizes expected risk-adjusted profit:

```text
maximize
    expected profit across scenarios
  - CVaR penalty for bad profit outcomes
  - imbalance risk penalty
  - battery degradation cost
```

---

# 6. Intraday decision problem

At an intraday stage, the already accepted day-ahead schedule is fixed.

The optimizer then chooses:

```text
q_ID[t]
```

for the remaining delivery intervals.

Again, the current intraday trade is a **here-and-now decision** based on the forecast distribution available at that intraday time.

Then the simulator advances, new forecasts arrive, and the model re-solves.

This gives you a clean way to evaluate:

```text
value of day-ahead commitment
value of intraday correction
value of better solar forecasts
value of better price forecasts
value of battery flexibility
value of probabilistic forecasting
```

---

# 7. Recommended Pyomo model type

For the first version, use an **extensive-form stochastic LP or MILP**.

Pyomo’s stochastic programming documentation describes the standard structure as a deterministic base model plus a scenario tree with uncertain parameters; the extensive form is then solved as one deterministic optimization problem. ([pysp.readthedocs.io][2])

For your use case, I would implement the extensive form manually in Pyomo first, rather than start with a full PySP implementation. It will be easier to debug and easier to connect to the simulator.

Use:

```text
LP if simultaneous charge/discharge is allowed but discouraged by efficiency and degradation
MILP if simultaneous charge/discharge must be strictly forbidden
```

HiGHS can solve LP and MIP models, so both are compatible. ([HiGHS][1])

---

# 8. Core mathematical formulation

Let:

```text
t ∈ T          time interval
ω ∈ Ω          forecast scenario
pω             probability of scenario ω
Δt             interval length, e.g. 0.25 hours
Sω,t           solar production in scenario ω
πDAω,t         day-ahead price in scenario ω
πIDω,t         intraday price in scenario ω
πsurplusω,t    surplus imbalance price
πshortfallω,t  shortfall imbalance price
```

Decision variables:

```text
qDA,t              day-ahead commitment
qID,ω,t            intraday correction
chargeω,t          battery charge power
dischargeω,t       battery discharge power
socω,t             battery state of charge
curtailω,t         curtailed solar
imb_posω,t         surplus imbalance
imb_negω,t         shortfall imbalance
```

Schedule:

```text
scheduleω,t = qDA,t + qID,ω,t
```

Physical injection:

```text
physicalω,t = Sω,t - curtailω,t + dischargeω,t - chargeω,t
```

Imbalance:

```text
physicalω,t - scheduleω,t = imb_posω,t - imb_negω,t
```

Battery state:

```text
socω,t+1 =
    socω,t
  + ηcharge * chargeω,t * Δt
  - dischargeω,t * Δt / ηdischarge
```

Scenario profit:

```text
profitω =
Σt Δt [
    πDAω,t * qDA,t
  + πIDω,t * qIDω,t
  + πsurplusω,t * imb_posω,t
  - πshortfallω,t * imb_negω,t
  - degradation_cost * (chargeω,t + dischargeω,t)
  - trading_fees
]
```

Expected profit:

```text
E[profit] = Σω pω * profitω
```

Risk-adjusted objective:

```text
maximize
    Σω pω * profitω
  - λCVaR * CVaRα(loss)
  - λimbalance * Σω Σt pω * (imb_posω,t + imb_negω,t)
```

Where:

```text
lossω = -profitω
```

CVaR can be formulated linearly, so it remains compatible with HiGHS.

---

# 9. CVaR formulation for downside risk

For confidence level `α`, for example `α = 0.95`, introduce:

```text
z
uω ≥ 0
```

with constraints:

```text
uω ≥ lossω - z
```

Then:

```text
CVaRα(loss) = z + 1 / (1 - α) * Σω pω * uω
```

The optimizer objective becomes:

```text
maximize expected_profit - λCVaR * CVaRα(loss)
```

This is useful because the battery may otherwise optimize for average profit while creating large downside exposure during bad solar forecast errors or price spikes.

---

# 10. Battery charge/discharge choice

There are two reasonable formulations.

## LP version, faster

Allow both `charge` and `discharge` variables but rely on degradation cost and efficiency losses to discourage simultaneous cycling.

Pros:

```text
fast
scales well with many scenarios
good for early research
compatible with large stochastic models
```

Cons:

```text
can occasionally create artificial simultaneous charge/discharge
especially under negative prices or unusual imbalance prices
```

## MILP version, stricter

Introduce binary variable:

```text
y_chargeω,t ∈ {0,1}
```

Constraints:

```text
chargeω,t    ≤ P_charge_max    * y_chargeω,t
dischargeω,t ≤ P_discharge_max * (1 - y_chargeω,t)
```

Pros:

```text
physically cleaner
prevents artificial battery cycling
```

Cons:

```text
much slower with many scenarios
scenario count may need to be reduced
```

My recommendation:

```text
MVP: use LP version
Validation model: use MILP version
Final research runs: compare LP vs MILP on reduced scenario sets
```

---

# 11. Pyomo/HiGHS implementation skeleton

Below is the kind of Pyomo model I would build first.

```python
import pyomo.environ as pyo


def build_stochastic_dispatch_model(data):
    """
    data should contain:
        T: list of time intervals
        OMEGA: list of scenario IDs
        prob[omega]
        solar[omega, t]
        price_da[omega, t]
        price_id[omega, t]
        price_imb_surplus[omega, t]
        price_imb_shortfall[omega, t]
        dt_hours
        battery parameters
        grid parameters
        risk parameters
    """

    m = pyo.ConcreteModel()

    # -----------------------
    # Sets
    # -----------------------
    m.T = pyo.Set(initialize=data["T"], ordered=True)
    m.O = pyo.Set(initialize=data["OMEGA"])

    T_list = list(data["T"])

    # -----------------------
    # Parameters
    # -----------------------
    m.prob = pyo.Param(m.O, initialize=data["prob"])

    m.solar = pyo.Param(
        m.O, m.T,
        initialize=lambda m, o, t: data["solar"][o, t]
    )

    m.price_da = pyo.Param(
        m.O, m.T,
        initialize=lambda m, o, t: data["price_da"][o, t]
    )

    m.price_id = pyo.Param(
        m.O, m.T,
        initialize=lambda m, o, t: data["price_id"][o, t]
    )

    m.price_imb_surplus = pyo.Param(
        m.O, m.T,
        initialize=lambda m, o, t: data["price_imb_surplus"][o, t]
    )

    m.price_imb_shortfall = pyo.Param(
        m.O, m.T,
        initialize=lambda m, o, t: data["price_imb_shortfall"][o, t]
    )

    dt = data["dt_hours"]
    eta_ch = data["eta_charge"]
    eta_dis = data["eta_discharge"]
    p_ch_max = data["battery_charge_mw"]
    p_dis_max = data["battery_discharge_mw"]
    soc_min = data["soc_min_mwh"]
    soc_max = data["soc_max_mwh"]
    soc_initial = data["soc_initial_mwh"]
    export_limit = data["export_limit_mw"]
    import_limit = data["import_limit_mw"]
    degradation_cost = data["degradation_cost_eur_per_mwh"]

    # -----------------------
    # First-stage DA decision
    # -----------------------
    m.q_da = pyo.Var(
        m.T,
        bounds=(-import_limit, export_limit)
    )

    # -----------------------
    # Scenario-dependent recourse
    # -----------------------
    m.q_id = pyo.Var(m.O, m.T, bounds=(-import_limit, export_limit))

    m.charge = pyo.Var(m.O, m.T, bounds=(0, p_ch_max))
    m.discharge = pyo.Var(m.O, m.T, bounds=(0, p_dis_max))
    m.soc = pyo.Var(m.O, m.T, bounds=(soc_min, soc_max))

    m.curtail = pyo.Var(m.O, m.T, within=pyo.NonNegativeReals)

    m.imb_pos = pyo.Var(m.O, m.T, within=pyo.NonNegativeReals)
    m.imb_neg = pyo.Var(m.O, m.T, within=pyo.NonNegativeReals)

    # -----------------------
    # Expressions
    # -----------------------
    def physical_expr(m, o, t):
        return (
            m.solar[o, t]
            - m.curtail[o, t]
            + m.discharge[o, t]
            - m.charge[o, t]
        )

    m.physical = pyo.Expression(m.O, m.T, rule=physical_expr)

    def schedule_expr(m, o, t):
        return m.q_da[t] + m.q_id[o, t]

    m.schedule = pyo.Expression(m.O, m.T, rule=schedule_expr)

    # -----------------------
    # Constraints
    # -----------------------
    def curtail_limit_rule(m, o, t):
        return m.curtail[o, t] <= m.solar[o, t]

    m.curtail_limit = pyo.Constraint(m.O, m.T, rule=curtail_limit_rule)

    def imbalance_rule(m, o, t):
        return (
            m.physical[o, t] - m.schedule[o, t]
            == m.imb_pos[o, t] - m.imb_neg[o, t]
        )

    m.imbalance_balance = pyo.Constraint(m.O, m.T, rule=imbalance_rule)

    def export_limit_rule(m, o, t):
        return m.physical[o, t] <= export_limit

    m.export_limit = pyo.Constraint(m.O, m.T, rule=export_limit_rule)

    def import_limit_rule(m, o, t):
        return m.physical[o, t] >= -import_limit

    m.import_limit = pyo.Constraint(m.O, m.T, rule=import_limit_rule)

    def soc_rule(m, o, t):
        idx = T_list.index(t)

        if idx == 0:
            soc_prev = soc_initial
        else:
            soc_prev = m.soc[o, T_list[idx - 1]]

        return m.soc[o, t] == (
            soc_prev
            + eta_ch * m.charge[o, t] * dt
            - m.discharge[o, t] * dt / eta_dis
        )

    m.soc_balance = pyo.Constraint(m.O, m.T, rule=soc_rule)

    # Optional terminal SoC condition
    if data.get("enforce_terminal_soc", True):
        final_t = T_list[-1]

        def terminal_soc_rule(m, o):
            return m.soc[o, final_t] >= data["soc_terminal_min_mwh"]

        m.terminal_soc = pyo.Constraint(m.O, rule=terminal_soc_rule)

    # Optional no-grid-charging condition
    if not data.get("allow_grid_charging", True):

        def no_grid_charging_rule(m, o, t):
            return m.charge[o, t] <= m.solar[o, t] - m.curtail[o, t]

        m.no_grid_charging = pyo.Constraint(
            m.O, m.T, rule=no_grid_charging_rule
        )

    # Optional strict no simultaneous charge/discharge
    if data.get("binary_battery_mode", False):
        m.y_charge = pyo.Var(m.O, m.T, within=pyo.Binary)

        def charge_binary_rule(m, o, t):
            return m.charge[o, t] <= p_ch_max * m.y_charge[o, t]

        def discharge_binary_rule(m, o, t):
            return m.discharge[o, t] <= p_dis_max * (1 - m.y_charge[o, t])

        m.charge_binary = pyo.Constraint(m.O, m.T, rule=charge_binary_rule)
        m.discharge_binary = pyo.Constraint(m.O, m.T, rule=discharge_binary_rule)

    # -----------------------
    # Profit expression
    # -----------------------
    def scenario_profit_rule(m, o):
        return sum(
            dt * (
                m.price_da[o, t] * m.q_da[t]
                + m.price_id[o, t] * m.q_id[o, t]
                + m.price_imb_surplus[o, t] * m.imb_pos[o, t]
                - m.price_imb_shortfall[o, t] * m.imb_neg[o, t]
                - degradation_cost * (m.charge[o, t] + m.discharge[o, t])
            )
            for t in m.T
        )

    m.scenario_profit = pyo.Expression(m.O, rule=scenario_profit_rule)

    m.expected_profit = pyo.Expression(
        expr=sum(m.prob[o] * m.scenario_profit[o] for o in m.O)
    )

    # -----------------------
    # Optional CVaR risk penalty
    # -----------------------
    lambda_cvar = data.get("lambda_cvar", 0.0)
    alpha = data.get("cvar_alpha", 0.95)

    if lambda_cvar > 0:
        m.z_cvar = pyo.Var()
        m.u_cvar = pyo.Var(m.O, within=pyo.NonNegativeReals)

        def cvar_constraint_rule(m, o):
            loss = -m.scenario_profit[o]
            return m.u_cvar[o] >= loss - m.z_cvar

        m.cvar_constraint = pyo.Constraint(m.O, rule=cvar_constraint_rule)

        m.cvar_loss = pyo.Expression(
            expr=m.z_cvar
            + (1.0 / (1.0 - alpha))
            * sum(m.prob[o] * m.u_cvar[o] for o in m.O)
        )

        objective_expr = m.expected_profit - lambda_cvar * m.cvar_loss

    else:
        objective_expr = m.expected_profit

    m.obj = pyo.Objective(expr=objective_expr, sense=pyo.maximize)

    return m
```

The solve step would look like this:

```python
from pyomo.contrib.appsi.solvers import Highs
from pyomo.contrib.appsi.base import TerminationCondition


def solve_with_highs(model, time_limit_seconds=60):
    solver = Highs()
    solver.config.time_limit = time_limit_seconds
    solver.highs_options["mip_rel_gap"] = 0.001

    results = solver.solve(model)

    if results.termination_condition != TerminationCondition.optimal:
        print("Warning: non-optimal termination:", results.termination_condition)

    return results
```

Pyomo’s APPSI solver interfaces are designed to be efficient when repeatedly resolving similar models with small changes, which is useful for rolling intraday optimization where only forecasts, prices, and current battery state change. ([Pyomo Documentation][3])

---

# 12. Important modelling correction: non-anticipativity

The simple model above indexes intraday decisions by scenario:

```text
qID[ω,t]
```

That is acceptable for a first **two-stage recourse approximation**, but a full multi-stage model should use **scenario-tree nodes**, not just scenarios.

Why?

Because decisions can only depend on information that has been revealed.

For example:

```text
D-1 day-ahead decision:
    all scenarios share q_DA[t]

After DA price is known:
    scenarios with the same observed DA outcome share q_IDA1[t]

After IDA1 information is known:
    scenarios with the same information share q_IDA2[t]

At delivery:
    dispatch may depend on realized solar and actual battery state
```

So the more correct structure is:

```text
q_DA[t]                root-level decision
q_IDA1[node, t]        decision at IDA1 information node
q_IDA2[node, t]        decision at IDA2 information node
q_IDA3[node, t]        decision at IDA3 information node
battery[node, t]       dispatch decision at relevant information node
```

For the MVP, I would not start with the full tree. I would do this instead:

```text
Version 1:
    two-stage extensive form

Version 2:
    rolling stochastic MPC

Version 3:
    explicit scenario tree with non-anticipativity constraints
```

This keeps the model build manageable.

---

# 13. Recommended simulator loop

```python
for delivery_day in simulated_days:

    world = true_world_generator.sample(delivery_day)

    # ----------------------
    # Day-ahead stage
    # ----------------------
    da_forecast = forecast_generator.make_scenarios(
        world=world,
        decision_time="D-1 before DA",
        n_scenarios=100
    )

    da_model = build_day_ahead_pyomo_model(
        forecast=da_forecast,
        current_soc=initial_soc,
        existing_schedule=None
    )

    solve_with_highs(da_model)

    q_da = extract_day_ahead_commitment(da_model)

    schedule = q_da.copy()

    # ----------------------
    # Intraday stages
    # ----------------------
    for stage in ["IDA1", "IDA2", "IDA3"]:

        id_forecast = forecast_generator.make_scenarios(
            world=world,
            decision_time=stage,
            n_scenarios=50,
            conditional_on_revealed_information=True
        )

        id_model = build_intraday_pyomo_model(
            forecast=id_forecast,
            current_soc=current_soc,
            existing_schedule=schedule,
            fixed_day_ahead_commitment=q_da
        )

        solve_with_highs(id_model)

        q_id = extract_intraday_trade(id_model)

        schedule += q_id

    # ----------------------
    # Delivery and settlement
    # ----------------------
    result = settle_realized_day(
        world=world,
        final_schedule=schedule,
        battery_policy="real_time_mpc"
    )

    results.append(result)
```

---

# 14. How the forecast generator should behave

At each later decision point, uncertainty should shrink.

For example:

```text
D-1 morning forecast:
    solar uncertainty wide
    price uncertainty wide

After DA market clears:
    DA price known
    ID price uncertainty still moderate
    solar uncertainty somewhat smaller

Intraday morning:
    solar uncertainty much smaller
    intraday price uncertainty smaller
    imbalance price uncertainty still material

Near delivery:
    solar forecast very sharp
    price/imbalance uncertainty remains partly unresolved
```

So the forecast generator should produce conditional scenarios:

```python
forecast_DA = generate_scenarios(decision_time="D-1 10:00")

forecast_IDA1 = generate_scenarios(
    decision_time="D-1 15:00",
    condition_on={"DA_price": realized_DA_price}
)

forecast_IDA2 = generate_scenarios(
    decision_time="D-1 22:00",
    condition_on={
        "DA_price": realized_DA_price,
        "updated_weather": realized_weather_signal
    }
)
```

---

# 15. Forecast model evaluation metrics

Since uncertainty is central, do not evaluate forecasts only using MAE or RMSE.

Use:

```text
coverage of prediction intervals
PIT histogram
CRPS
pinball loss for quantiles
energy score for multivariate scenarios
scenario calibration by hour and season
profit-weighted forecast value
regret versus oracle
value of probabilistic forecast versus point forecast
```

The most important business metric is not:

```text
Was the forecast numerically accurate?
```

It is:

```text
Did the forecast distribution lead to better trading and dispatch decisions?
```

---

# 16. Recommended implementation phases

## Phase 1: deterministic Pyomo model

Inputs:

```text
one price forecast path
one solar forecast path
```

Purpose:

```text
debug signs
debug settlement
debug battery constraints
debug day-ahead and intraday accounting
```

Solver:

```text
Pyomo + HiGHS LP
```

## Phase 2: scenario-based stochastic model

Inputs:

```text
20 to 50 scenarios
scenario probabilities
solar scenario paths
DA price scenario paths
ID price scenario paths
imbalance price scenario paths
```

Purpose:

```text
replace point forecast with expected risk-adjusted profit
```

Solver:

```text
Pyomo + HiGHS LP
```

## Phase 3: risk-aware stochastic model

Add:

```text
CVaR
imbalance penalty
battery degradation
forecast calibration diagnostics
```

Solver:

```text
Pyomo + HiGHS LP
```

## Phase 4: physical battery refinement

Add:

```text
binary charge/discharge mode
export/import limits
optional grid charging
terminal SoC rule
curtailment
availability outages
```

Solver:

```text
Pyomo + HiGHS MILP
```

## Phase 5: rolling intraday stochastic MPC

Add:

```text
DA stage
IDA1 stage
IDA2 stage
IDA3 stage
real-time battery dispatch
conditional scenario updates
```

Solver:

```text
repeated Pyomo/HiGHS solves
prefer APPSI interface for repeated solves
```

## Phase 6: full scenario tree

Add:

```text
multi-stage non-anticipativity
node-based decisions
scenario reduction
possibly decomposition if model becomes large
```

---

# 17. My recommended MVP

For the first serious prototype, I would build this:

```text
Resolution:
    15 minutes

Markets:
    day-ahead + one intraday correction

Forecast input:
    50 probabilistic scenarios with probabilities

Optimizer:
    Pyomo extensive-form stochastic LP

Solver:
    HiGHS through Pyomo APPSI

Battery:
    linear charge/discharge model
    degradation cost
    export limit
    optional grid charging

Risk:
    expected profit first
    then add CVaR

Outputs:
    profit distribution
    realized profit
    imbalance
    battery cycles
    intraday value
    forecast value
    regret versus oracle
```

Then extend to:

```text
multiple intraday stages
rolling stochastic MPC
MILP battery mode
full scenario tree
CVaR risk control
scenario reduction
Estonia-calibrated price and solar forecast errors
```

The most important design decision is: **make the forecast object probabilistic from day one**. The Pyomo model should consume scenarios and probabilities directly, and point forecasts should only be used as baselines.

[1]: https://highs.dev/ "HiGHS - High-performance parallel linear optimization software"
[2]: https://pysp.readthedocs.io/en/latest/pysp.html "Stochastic Programming — PySP 6.0.0 documentation"
[3]: https://pyomo.readthedocs.io/en/6.8.0/library_reference/appsi/appsi.html "APPSI — Pyomo 6.8.0 documentation"
