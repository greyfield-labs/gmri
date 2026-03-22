# Macro Regime Index

Classifies the US economy into one of four macro regimes on a monthly basis using public FRED data, then visualizes the full history back to 1970.

## Regimes

| Regime | Growth | Inflation |
|---|---|---|
| **Goldilocks** | Above trend | Below 2.5% |
| **Overheating** | Above trend | Above 2.5% |
| **Stagflation** | Below trend | Above 2.5% |
| **Deflationary Bust** | Below trend | Below 2.5% |

## Methodology

### Data Sources (FRED API)

| Series | Description | Frequency |
|---|---|---|
| `CPIAUCSL` | Consumer Price Index | Monthly |
| `UNRATE` | Unemployment Rate | Monthly |
| `GDPC1` | Real GDP (chained 2017 dollars) | Quarterly → interpolated monthly |
| `NROU` | Natural Rate of Unemployment (NAIRU) | Quarterly → interpolated monthly |
| `GDPPOT` | Real Potential GDP | Quarterly → interpolated monthly |
| `USREC` | NBER Recession Indicator | Monthly |

### Classification Logic

1. **CPI YoY**: computed as 12-month percent change of `CPIAUCSL`
2. **GDP Gap**: `(Real GDP − Potential GDP) / Potential GDP × 100`, then smoothed with a 3-month rolling average to reduce noise from quarterly interpolation
3. **Unemployment Gap**: `UNRATE − NROU` (positive = labor market slack; negative = tight labor market)
4. **Growth above trend**: `GDP Gap > 0` OR `Unemployment Gap < 0` (either signal is sufficient)
5. **Inflation above threshold**: `CPI YoY > 2.5%`
6. Regime is assigned from the 2×2 matrix of these two binary conditions

### Quarterly-to-Monthly Interpolation

- `GDPC1` and `GDPPOT`: cubic spline interpolation (captures business cycle curvature)
- `NROU`: linear interpolation (slow-moving structural estimate; cubic can oscillate)

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure FRED API key
cp .env.example .env
# Edit .env and set your FRED_API_KEY

# 3. Run
python main.py
```

A free FRED API key can be obtained at https://fred.stlouisfed.org/docs/api/api_key.html

## Outputs

| File | Description |
|---|---|
| `regime_timeline.png` | Colored regime bands across 1970–present with NBER recession shading and CPI overlay |
| `transition_matrix.png` | Heatmap of month-over-month regime transition probabilities |
| `signal_chart.png` | Two-panel chart: GDP output gap (raw + smoothed) and CPI YoY with 2.5% threshold |

Summary statistics (% time in each regime, average spell duration, number of spells) are printed to stdout.

## Project Structure

```
├── main.py          Orchestration entry point
├── data.py          FRED data fetching and preprocessing
├── classifier.py    Regime classification logic
├── charts.py        Visualization
├── requirements.txt
├── .env             FRED API key (not committed)
└── .env.example     Template
```
