# quant-sims

A progressive collection of Monte Carlo and stochastic process simulations, moving from basic probability/betting problems toward classic quantitative finance models.

The goal of this project is to build intuition for core quant finance concepts (edge, position sizing, risk of ruin, stochastic processes, option pricing, tail risk) by implementing them from scratch, visualizing the results, and comparing simulated behavior against theory.

## Structure

```
quant-sims/
├── src/quant_sims/          # core simulation code (importable package)
│   ├── betting/              # coin flip games, Kelly criterion
│   ├── stochastic_processes/ # random walk, Brownian motion, GBM
│   └── utils/                 # shared plotting / helper functions
├── notebooks/                # narrative walkthroughs, visualizations, findings
├── tests/                     # unit tests for core logic
└── docs/                      # theory notes, derivations
```

Simulation logic lives in `src/`; notebooks import from there rather than duplicating logic, so results stay reproducible and testable.

## Roadmap

### Phase 1 — Betting & Position Sizing
- [ ] Simple coin flip simulation (biased/unbiased, single trial)
- [ ] Repeated betting with fixed capital fraction
- [ ] Kelly Criterion — optimal fraction derivation + simulation
- [ ] Compare strategies: fixed fraction vs. Kelly vs. half-Kelly vs. martingale vs. all-in
- [ ] Monte Carlo over many simulated paths — distribution of outcomes, equity curves
- [ ] Gambler's ruin — probability of ruin as a function of bet size and edge

### Phase 2 — Stochastic Processes
- [ ] Simple random walk
- [ ] Convergence of random walk to Brownian motion (Wiener process)
- [ ] Geometric Brownian Motion (GBM) — simulating stock price paths
- [ ] Comparing simulated GBM paths to real market log-returns (fat tails discussion)

### Phase 3 — Option Pricing
- [ ] Black-Scholes closed-form pricing
- [ ] Monte Carlo option pricing (European options) and convergence to Black-Scholes
- [ ] Greeks via finite differences

### Phase 4 — Risk Modeling
- [ ] Value at Risk (VaR) and Expected Shortfall via Monte Carlo
- [ ] Extreme Value Theory (EVT) for tail risk estimation

### Phase 5 — Advanced Models
- [ ] Mean-reverting rate models: Vasicek / CIR
- [ ] Jump diffusion (Merton model) — adding jumps to GBM
- [ ] Stochastic volatility (Heston model)

## Setup

```bash
git clone https://github.com/<your-username>/quant-sims.git
cd quant-sims
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running

```bash
# run a simulation module directly
python -m quant_sims.betting.coin_flip

# or explore results interactively
jupyter notebook notebooks/
```

## Tests

```bash
pytest tests/
```

## License

MIT
