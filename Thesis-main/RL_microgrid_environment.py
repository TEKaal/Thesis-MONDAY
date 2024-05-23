import matplotlib.pyplot as plt
from RL_read_energy_data import *
from RL_read_solar_data import *
from RL_read_wind_data import *
from RL_read_grid_costs import *

timesteps = 35040

import random
import numpy as np
import pandas as pd

np.random.seed(0)

from pymgrid import Microgrid
from pymgrid.modules import (
    BatteryModule,
    LoadModule,
    RenewableModule,
    GridModule)
from pymgrid.forecast import OracleForecaster


def monday_interval_ranges_from_offset(start_day_offset, total_intervals=35040, intervals_per_day=96):
    start_index = start_day_offset * intervals_per_day
    intervals_per_week = 7 * intervals_per_day
    monday_indices = np.arange(start_index, total_intervals, intervals_per_week)
    # Create tuple ranges for each Monday
    monday_ranges = [(index, index + intervals_per_day) for index in monday_indices]
    return monday_ranges

def calculate_wind_power(wind_speeds, radius, power_coefficient, air_density=1.225):
    """
    Calculate average wind power generation in kilowatts based on wind speeds.

    Parameters:
        wind_speeds (np.array): Array of wind speeds in m/s.
        radius (float): Radius of the wind turbine rotor in meters.
        power_coefficient (float): Coefficient of performance of the turbine (typically between 0.35 and 0.45).
        air_density (float): Density of air in kg/m^3 (approximately 1.225 kg/m^3 at sea level).

    Returns:
        np.array: Power generated in kilowatts for each interval.
    """
    # Area of the rotor swept by the blades (πr^2)
    area = np.pi * (radius ** 2)

    # Calculate power in watts using the wind power formula: P = 0.5 * ρ * A * v^3 * Cp
    power_watts = 0.5 * air_density * area * (wind_speeds ** 3) * power_coefficient

    # Convert power from watts to kilowatts
    power_kilowatts = power_watts / 1000
    power_kilowattshour = power_kilowatts * 0.25

    return power_kilowattshour

def adjust_load_profile(load_profile, steps):
    length = len(load_profile)

    if len(load_profile) > timesteps:
        # Keep only the last 'timesteps' values (warm up period taken into account)
        adjusted_profile = load_profile[-timesteps:]
    elif length < timesteps:
        # Extend the load profile if it's shorter than required
        # This example pads with the last value, but other strategies can be used
        padding = [load_profile[-1]] * (timesteps - length)
        adjusted_profile = np.concatenate((load_profile, padding))
    else:
        # If the load profile already matches the timesteps, use it as is
        adjusted_profile = load_profile

    return adjusted_profile[:steps]

def create_microgrid(Energy_consumption, combined_df, df_buildings, steps=35040):
    load_modules = {}
		
    # Access the first row assuming it contains the grid information
    grid_info = combined_df.iloc[len(df_buildings)]  # Use iloc to access by position
    horizon = int(grid_info['Horizon'])

    monday_ranges = monday_interval_ranges_from_offset(2, steps, 96)

    # Process load data for each house, filtering for Mondays
    for house_id, load_profile in Energy_consumption.items():
        # Concatenate all Monday data for the year into a single array
        all_mondays_load = np.concatenate([load_profile[start:end] for start, end in monday_ranges])
        load_module = LoadModule(time_series=all_mondays_load)
        load_module.set_forecaster(forecaster="oracle", forecast_horizon=horizon)
        load_modules[f'load_{house_id}'] = load_module

    # Filter out the rows where the Type is "Battery"
    battery_df = combined_df[combined_df["Type"] == "Battery"]
    battery_modules = {}

    for index, row in battery_df.iterrows():
        # Create a BatteryModule instance for each battery row
        battery_module = BatteryModule(
            min_capacity=row["Min_capaci"],  # Check the exact column name in your DataFrame
            max_capacity=row["Max_capaci"],  # Check the exact column name in your DataFrame
            max_charge=row["Max_charge"],  # Check the exact column name in your DataFrame
            max_discharge=row["Max_discha"],  # Check the exact column name in your DataFrame
            efficiency=row["Efficiency"],  # Check the exact column name in your DataFrame
            init_soc=random.uniform(0, 1),  # Initialize SOC randomly between 0 and 1
            battery_cost_cycle= 0 #row["Battery_co"]  # Check the exact column name in your DataFrame
        )

        # Store the battery module in the dictionary with its index as the key
        battery_modules[index] = battery_module

    solar_data_array = solar_data(timesteps, steps)
    all_mondays_solar_data = np.concatenate([solar_data_array[start:end] for start, end in monday_ranges])

    # Convert the column to numeric, coercing errors to NaN
    df_buildings["TNO_dakopp_m2"] = pd.to_numeric(df_buildings["TNO_dakopp_m2"], errors='coerce')

    # TNO_p_dak_horizontaal
    # Sum the column, automatically skipping NaN values
    # lets say 80 percent is available
    roof_partition = df_buildings["TNO_dakopp_m2"].sum() * 0.6

    # Print the calculated sum
    print("Horizontal roof partition calculated:", roof_partition)

    # problem is not all buildings have this
    efficiency_pv = 0.9
    solar_energy = roof_partition * all_mondays_solar_data * 0.25 * efficiency_pv

    # Scaling the solar energy by an arbitrary factor (like 10000 here) for your application needs
    solar_energy = 10000 * solar_energy

    solar_energy = RenewableModule(time_series=solar_energy)
    solar_energy.set_forecaster(forecaster="oracle",
                                             forecast_horizon=horizon,
                                             forecaster_increase_uncertainty=False,
                                             forecaster_relative_noise=False)


    # Filter out the rows where the Type is "Windturbine"
    windturbine_df = combined_df[combined_df["Type"] == "Windturbine"]
    # Dictionary to store wind modules
    wind_modules = {}

    # Iterate through each wind turbine entry
    for index, row in windturbine_df.iterrows():
        # Fetch wind data and process it for each Monday
        wind_speed_array = wind_data(timesteps, steps) 
        all_mondays_wind_speeds = np.concatenate([wind_speed_array[start:end] for start, end in monday_ranges])
        wind_data_array = calculate_wind_power(all_mondays_wind_speeds, row["Radius_WT"], row["Power_coef"])

        # Scaling the wind power production by a factor (e.g., 40 here) as per your existing setup
        wind_power_scaled = 40 * wind_data_array
        
        # Create and configure the wind module for each turbine
        wind_module = RenewableModule(time_series=wind_power_scaled)
        wind_module.set_forecaster(forecaster="oracle",
                                   forecast_horizon=horizon,  # Adjust according to your system's forecast needs
                                   forecaster_increase_uncertainty=False,
                                   forecaster_relative_noise=False)

        # Store each wind module in the dictionary with a unique key based on turbine index
        wind_modules[f'windmod_{index}'] = wind_module

    max_import = grid_info['Max_import']
    max_export = grid_info['Max_export']
    co2_price = grid_info['CO2_price']

    # Now you can use these values in your application
    print("Max Import:", max_import)
    print("Max Export:", max_export)
    print("CO2 Price:", co2_price)

    import_array = interpolate_import_costs(timesteps, steps)
    all_mondays_grid_import = np.concatenate([import_array[start:end] for start, end in monday_ranges])
	
    # export_array = interpolate_export_costs(timesteps) # https://www.zonnepanelen-info.nl/blog/stappenplan-stroom-terugleveren-aan-het-net/ per kwh

    #import, export, Co2
    #CO2 NOG VERVANGEN
    grid_ts = [0.2,0.1,co2_price] * np.ones(((96*52), 3))
    grid_ts[:, 0] = all_mondays_grid_import
    grid_ts[:, 1] = all_mondays_grid_import # just
    grid_ts[:, 0] = np.where(grid_ts[:, 0] < 0, 0, grid_ts[:, 0])
    grid_ts[:, 1] = np.where(grid_ts[:, 1] < 0, 0, grid_ts[:, 1])

    # combined_df["Type"]=="Grid" NOG VERVANGEN
    grid = GridModule(max_import=max_import,
                      max_export=max_export,
                      time_series=grid_ts)

    grid.set_forecaster(forecaster="oracle",
                                   forecast_horizon=horizon,
                                   forecaster_increase_uncertainty=False,
                                   forecaster_relative_noise=False)

    modules = [grid,
               ('solar_energy', solar_energy)]

    combined_modules = modules + list(load_modules.values()) + list(battery_modules.values()) + list(wind_modules.values())
    microgrid = Microgrid(combined_modules)

    return microgrid





