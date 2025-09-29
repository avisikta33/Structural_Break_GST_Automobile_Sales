import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

def fit_sarimax_with_intervention(data, series_name, intervention_date, 
                                  order=(1, 1, 1), seasonal_order=(1, 1, 0, 12)):
    """
    Fits a SARIMAX model with a dummy intervention variable.

    Args:
        data (pd.DataFrame): DataFrame containing the time series data. Must have a DatetimeIndex.
        series_name (str): Name of the column representing the sales series.
        intervention_date (str): Date string (e.g., 'YYYY-MM-DD') for the intervention.
        order (tuple): Non-seasonal (p, d, q) order of the model.
        seasonal_order (tuple): Seasonal (P, D, Q, S) order of the model.

    Returns:
        tuple: Fitted SARIMAX model and a dictionary of intervention effects.
    """
    print(f"\n--- Fitting SARIMAX model for {series_name} with intervention at {intervention_date} ---")

    y = data[series_name]

    # Create intervention variable (dummy variable for GST regime)
    intervention_var = (data.index >= intervention_date).astype(int)
    
    # Add a constant term to exogenous variables
    exog = sm.add_constant(intervention_var)

    # Split data into training and testing sets (e.g., 80/20 split)
    train_size = int(len(y) * 0.8)
    train_y, test_y = y[:train_size], y[train_size:]
    train_exog, test_exog = exog[:train_size], exog[train_size:]

    print(f"  Training SARIMAX on {len(train_y)} observations...")
    print(f"  Testing on {len(test_y)} observations...")

    model = SARIMAX(train_y, exog=train_exog,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False)
    
    try:
        model_fit = model.fit(disp=False) # disp=False to suppress iteration output
        print("  SARIMAX Model Summary:")
        print(model_fit.summary())
    except Exception as e:
        print(f"  Error fitting SARIMAX model: {e}")
        return None, {'impact': 'Error during fitting'}

    # Get the coefficient for the intervention variable
    # The name will typically be 'intervention_var' if added directly, or 'x1' if using sm.add_constant
    # Check model_fit.params for the exact name
    intervention_param_name = None
    for name in model_fit.params.index:
        if 'intervention_var' in name or 'x1' in name: # 'x1' if intervention_var is the second column after constant
            intervention_param_name = name
            break
            
    intervention_effect = None
    if intervention_param_name:
        intervention_effect = model_fit.params[intervention_param_name]
        print(f"\n  Estimated Intervention (GST) Impact: {intervention_effect:.4f}")
        print(f"  P-value for Intervention Impact: {model_fit.pvalues[intervention_param_name]:.4f}")
        if model_fit.pvalues[intervention_param_name] < 0.05:
            print("  Conclusion: The intervention (GST) has a statistically significant impact.")
        else:
            print("  Conclusion: The intervention (GST) does NOT have a statistically significant impact.")
    else:
        print("\n  Could not find intervention variable coefficient in model parameters.")

    # Forecast and evaluate on test set
    forecast_results = model_fit.get_forecast(steps=len(test_y), exog=test_exog)
    forecast = forecast_results.predicted_mean
    conf_int = forecast_results.conf_int()

    rmse = np.sqrt(mean_squared_error(test_y, forecast))
    print(f"  RMSE on test set: {rmse:.4f}")

    return model_fit, {'impact': intervention_effect, 'rmse': rmse, 'forecast': forecast, 'conf_int': conf_int}


def plot_sarimax_results(data, series_name, model_fit, forecast_data, intervention_date):
    """
    Plots the original series, fitted values, and forecast from the SARIMAX model.

    Args:
        data (pd.DataFrame): Original DataFrame.
        series_name (str): Name of the sales series.
        model_fit: Fitted SARIMAX model object.
        forecast_data (dict): Dictionary containing 'forecast' and 'conf_int' from SARIMAX results.
        intervention_date (str): Date string for the intervention.
    """
    print(f"\n--- Generating plot for {series_name} SARIMAX results ---")

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot original series
    ax.plot(data.index, data[series_name], label='Original Sales', color='blue', alpha=0.7)

    # Plot fitted values from the training period
    # Ensure exog for prediction matches training exog
    train_size = len(model_fit.fittedvalues) # This is the length of the training data used for fitting
    
    # Re-create the intervention variable for the full data and use it to predict the fitted values
    intervention_var_full = (data.index >= intervention_date).astype(int)
    exog_full = sm.add_constant(intervention_var_full)
    
    # Get predictions for the entire dataset using the fitted model and full exogenous variables
    # This will give 'fitted' values for training and 'predicted' values for testing if needed
    # However, model_fit.predict() on the training data without exog returns fittedvalues.
    # For accurate visualization, it's best to predict on the *training* exog for in-sample.
    
    # Use model_fit.predict() with the training exog for in-sample predictions
    # And then the actual forecast for the test set
    
    # In-sample predictions (fitted values)
    # The fittedvalues property is already available from the model_fit object
    fitted_values_index = data.index[:train_size]
    ax.plot(fitted_values_index, model_fit.fittedvalues, label='SARIMAX Fitted', color='green', linestyle='--')

    # Plot forecast
    forecast_index = forecast_data['forecast'].index
    ax.plot(forecast_index, forecast_data['forecast'], label='SARIMAX Forecast', color='red')
    ax.fill_between(forecast_index,
                    forecast_data['conf_int'].iloc[:, 0],
                    forecast_data['conf_int'].iloc[:, 1], color='red', alpha=0.1)

    # Add intervention line
    ax.axvline(pd.to_datetime(intervention_date), color='purple', linestyle=':', label='GST Implementation')

    ax.set_title(f'SARIMAX Model Fit and Forecast for {series_name}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Sales Volume')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()
    print("  Plot generated.")