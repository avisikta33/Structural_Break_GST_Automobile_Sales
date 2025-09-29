import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_sales_data(start_date, end_date, base_sales, trend, seasonality_amplitude, noise_std, gst_date=None, gst_impact_ev=0, gst_impact_suv=0):
    dates = pd.date_range(start=start_date, end=end_date, freq='MS')
    n_months = len(dates)

    # Base sales with trend and seasonality
    time_index = np.arange(n_months)
    base = base_sales + trend * time_index
    seasonality = seasonality_amplitude * np.sin(2 * np.pi * time_index / 12) # 12-month cycle

    ev_sales = base * 0.1 + seasonality * 0.5 + np.random.normal(0, noise_std, n_months)
    suv_sales = base * 0.4 + seasonality * 0.8 + np.random.normal(0, noise_std, n_months)
    two_wheeler_sales = base * 0.5 + seasonality * 1.0 + np.random.normal(0, noise_std, n_months)

    # Introduce GST impact
    if gst_date:
        gst_index = dates.get_loc(gst_date, method='nearest')
        
        # Immediate jump + sustained growth (example for EV)
        ev_sales[gst_index:] *= (1 + gst_impact_ev)
        suv_sales[gst_index:] *= (1 + gst_impact_suv)
        
        # For two-wheelers, assume no significant break or even a slight negative initial impact that recovers
        # Or no change as per the project description
        # two_wheeler_sales[gst_index:] *= (1 + gst_impact_tw) 

    # Ensure no negative sales
    ev_sales = np.maximum(0, ev_sales)
    suv_sales = np.maximum(0, suv_sales)
    two_wheeler_sales = np.maximum(0, two_wheeler_sales)

    df = pd.DataFrame({
        'Date': dates,
        'EV_Sales': ev_sales,
        'SUV_Sales': suv_sales,
        'Two_Wheeler_Sales': two_wheeler_sales
    })
    df = df.set_index('Date')
    return df

if __name__ == '__main__':
    start_date = '2015-01-01'
    end_date = '2022-12-31'
    base_sales = 1000
    trend = 5
    seasonality_amplitude = 200
    noise_std = 50
    gst_date = '2017-07-01' # July 2017
    gst_impact_ev = 0.50 # +50% relative impact on EV sales post-GST
    gst_impact_suv = 0.238 # +23.8% impact on SUV sales post-GST

    sales_df = generate_sales_data(start_date, end_date, base_sales, trend, seasonality_amplitude, noise_std, gst_date, gst_impact_ev, gst_impact_suv)
    sales_df.to_csv('../data/synthetic_automobile_sales.csv')
    print("Synthetic data generated and saved to data/synthetic_automobile_sales.csv")