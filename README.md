# GTA VI Trophy Rarity Forecast

[![Bounty](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![Rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/mcukimay/makefun-gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![Reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/mcukimay/makefun-gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

Entry for make.fun's [gta6-trophy-rarity-forecast-01](https://make.fun/bounty/gta6-trophy-rarity-forecast-01):
for each of GTA VI's 55 PS5 trophies, five quantiles (5/25/50/75/95%) of its Exophase earn rate at the
case target, scored by pinball loss on the logit scale. Node 22, standard library only, offline,
deterministic; run with `MAKEFUN_INPUT_DIR` / `MAKEFUN_OUTPUT_DIR` set (defaults `/input`, `/output`).

## Model

Exophase only changes the page at its own refreshes, so a case's reads collapse to distinct page
states; everything is fitted on those states, on the case's own data alone.

- **Players and earners saturate.** `players_tracked` and each trophy's earners
  (rate × players / 100) are fitted with `A − B·exp(−t/τ)` over a grid of time constants τ.
  A trophy's τ is shrunk toward its tier's pooled τ (thin series borrow strength), and its earners
  ceiling is capped by the fitted players ceiling — so player-pool dilution and earner growth are
  forecast separately instead of conflating them in the rate.
- **Ensemble median.** In logit space the median blends the earners/players decomposition (0.25)
  with two damped `logit(rate) ~ a + b·ln(1+t)` trends — the last day of states (0.55) and all
  states (0.20) — each trend's extension past the last read damped by 0.85 so a hot recent slope
  cannot run away over the 4.5-day horizon.
- **Honest spread.** σ per trophy grows with √horizon, the earners fit's residual scaled by
  extrapolation distance, the shrinkage weight, and the ensemble's own disagreement; quantiles are
  `inv_logit(median + z·σ)`, kept ordered and inside (0, 100).
- **Fallbacks.** A trophy with too few distinct states takes its last rate plus the pooled logit
  growth of trophies in its rate band, with a wider spread.

Constants were calibrated on simulated launch weeks that reproduce the documented process
(step refreshes, a saturating player pool, per-trophy earner accumulation), where this model
scores ≈ 0.22 mean logit pinball against ≈ 0.48 for persistence and ≈ 0.31 for a recent-curve
extension. On the kit's frozen practice histories the program stays valid and near the last read,
as any reader of those cases must.
