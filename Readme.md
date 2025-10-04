# Automobile Sales Structural Break Detection

## 📊 Project Overview

This project implements comprehensive structural break detection methods to analyze the impact of GST (Goods and Services Tax) implementation on the Indian automobile industry. The analysis focuses on three vehicle segments: Electric Vehicles (EVs), SUVs, and Two-Wheelers, examining how the July 2017 GST implementation created structural breaks in their sales patterns.

## 🎯 Key Features

- **Synthetic Data Generation**: Creates realistic automobile sales data with configurable GST impact parameters
- **Multiple Structural Break Tests**: 
  - Chow Test (known break point detection)
  - CUSUM Test (unknown break detection)
  - Bai-Perron Test (multiple break points detection)
- **SARIMAX Modeling**: Time series forecasting with intervention analysis
- **Visualization**: Comprehensive plotting of test results and model forecasts

## 📁 Project Structure

```
automobile-sales-analysis/
│
├── data/
│   └── synthetic_automobile_sales.csv  # Generated sales data
│
├── src/
│   ├── data.generator.py              # Synthetic data generation
│   ├── break.detector.py              # Structural break detection tests
│   └── Sarimax.py                     # SARIMAX modeling with intervention
│
├── notebooks/                          # Jupyter notebooks for analysis
├── results/                           # Output plots and test results
└── README.md                          # Project documentation
```

## 🛠️ Installation

### Prerequisites

- Python 3.7+
- pip package manager

### Required Libraries

```bash
pip install numpy pandas matplotlib seaborn
pip install statsmodels scikit-learn scipy
```

### Optional Libraries

For advanced break detection (Bai-Perron test):
```bash
pip install ruptures  # For changepoint detection
```

## 🚀 Quick Start

### 1. Generate Synthetic Data

```python
from data.generator import generate_sales_data

# Generate automobile sales data with GST impact
sales_df = generate_sales_data(
    start_date='2015-01-01',
    end_date='2022-12-31',
    base_sales=1000,
    trend=5,
    seasonality_amplitude=200,
    noise_std=50,
    gst_date='2017-07-01',
    gst_impact_ev=0.50,      # 50% increase for EVs
    gst_impact_suv=0.238      # 23.8% increase for SUVs
)
```

### 2. Perform Structural Break Detection

```python
from break.detector import perform_chow_test, perform_cusum_test, perform_bai_perron_test
import pandas as pd

# Load data
data = pd.read_csv('data/synthetic_automobile_sales.csv', index_col='Date', parse_dates=True)

# Chow Test (for known break point)
chow_results = perform_chow_test(data, 'EV_Sales', '2017-07-01')

# CUSUM Test (for unknown breaks)
cusum_values, upper_bound, lower_bound, dates = perform_cusum_test(data, 'EV_Sales')

# Bai-Perron Test (for multiple breaks)
break_points = perform_bai_perron_test(data, 'EV_Sales', max_breaks=5)
```

### 3. SARIMAX Modeling with Intervention

```python
from Sarimax import fit_sarimax_with_intervention, plot_sarimax_results

# Fit SARIMAX model with GST intervention
model_fit, results = fit_sarimax_with_intervention(
    data=data,
    series_name='EV_Sales',
    intervention_date='2017-07-01',
    order=(1, 1, 1),
    seasonal_order=(1, 1, 0, 12)
)

# Visualize results
plot_sarimax_results(data, 'EV_Sales', model_fit, results, '2017-07-01')
```

## 📊 Methodology

### Data Generation (`data.generator.py`)

The synthetic data generator creates realistic automobile sales patterns with:
- **Base sales**: Starting sales volume
- **Trend**: Linear growth over time
- **Seasonality**: 12-month cyclical patterns
- **GST Impact**: Configurable structural break at GST implementation
- **Noise**: Random variations to simulate market volatility

### Structural Break Detection (`break.detector.py`)

#### 1. **Chow Test**
- Tests for structural break at a known point (GST implementation date)
- Compares regression models before and after the break
- Returns F-statistic and p-value for hypothesis testing

#### 2. **CUSUM Test**
- Detects unknown structural breaks
- Uses cumulative sum of recursive residuals
- Visual inspection of CUSUM values against critical bounds

#### 3. **Bai-Perron Test**
- Identifies multiple structural breaks
- Conceptual implementation (requires external libraries for full functionality)
- Returns list of detected break dates

### SARIMAX Modeling (`Sarimax.py`)

- **Model**: Seasonal ARIMA with exogenous variables
- **Intervention Variable**: Binary dummy for pre/post GST periods
- **Features**:
  - Automatic model fitting with configurable ARIMA orders
  - Intervention impact quantification
  - Out-of-sample forecasting
  - Confidence interval generation
  - Comprehensive visualization

## 📈 Expected Results

### Key Findings

1. **EV Sales**: Significant positive structural break (~50% increase post-GST)
2. **SUV Sales**: Moderate positive structural break (~23.8% increase post-GST)
3. **Two-Wheeler Sales**: No significant structural break detected

### Statistical Significance

- Chow Test p-values < 0.05 indicate significant breaks
- CUSUM values exceeding critical bounds suggest instability
- SARIMAX intervention coefficients quantify impact magnitude

## 🎯 Use Cases

- **Policy Impact Analysis**: Evaluate GST effectiveness on different vehicle segments
- **Market Forecasting**: Predict sales considering structural changes
- **Investment Decisions**: Identify growth segments post-policy changes
- **Academic Research**: Study structural breaks in economic time series

## 📝 Configuration

### Customizing GST Impact

Modify the impact parameters in `data.generator.py`:

```python
gst_impact_ev = 0.50    # 50% increase for EVs
gst_impact_suv = 0.238  # 23.8% increase for SUVs
```

### Adjusting SARIMAX Parameters

Configure model orders for different data characteristics:

```python
order = (p, d, q)                    # Non-seasonal ARIMA order
seasonal_order = (P, D, Q, S)        # Seasonal ARIMA order
```

## ⚠️ Important Notes

1. **Bai-Perron Test**: The current implementation provides conceptual output. For full functionality, install specialized packages like `ruptures`.

2. **CUSUM Test**: Visual inspection is recommended alongside statistical interpretation.

3. **Data Requirements**: Ensure sufficient observations before and after break points for reliable test results.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for:
- Bug fixes
- Additional structural break tests
- Improved visualization
- Documentation enhancements

## 📚 References

- Chow, G. C. (1960). "Tests of Equality Between Sets of Coefficients in Two Linear Regressions"
- Brown, R. L., Durbin, J., & Evans, J. M. (1975). "Techniques for Testing the Constancy of Regression Relationships Over Time"
- Bai, J., & Perron, P. (2003). "Computation and Analysis of Multiple Structural Change Models"

## 👥 Authors

**Souraj Chakraborty** & **Avisikta Das**

## 📧 Contact

For questions or collaboration opportunities, please reach out to the authors souraj.bhu.stats26@gmail.com & avisikta.bhu.stats26@gmail.com.

---

**Note**: This project uses synthetic data for demonstration purposes. For real-world applications, replace with actual automobile sales data from reliable sources.
