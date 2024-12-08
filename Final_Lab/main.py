import numpy as np
import matplotlib.pyplot as plt

# Parameters (can be adjusted for scenario analysis)
preindustrial_co2 = 280  # ppm
climate_sensitivity = 5.35  # W/m^2 for doubling CO2
lambda_sensitivity = 0.8  # Temperature sensitivity factor (C/W/m^2)
initial_atmosphere = 850  # Initial atmospheric CO2 (ppm)
initial_ocean = 38000  # Initial oceanic CO2 reservoir (GtC)
initial_terrestrial = 2000  # Initial terrestrial CO2 reservoir (GtC)
emissions_rate = 10  # Anthropogenic emissions (GtC/year)

# Flux coefficients (adjust for feedback mechanisms)
k_ao = 0.2  # Atmosphere to Ocean
k_oa = 0.1  # Ocean to Atmosphere
k_at = 0.1  # Atmosphere to Terrestrial
k_ta = 0.05  # Terrestrial to Atmosphere

# Feedback strength
feedback_strength = 0.02  # Adjusts k_ao and k_at based on warming

def co2_flux(y, emissions_rate, temperature):
    # Reservoir states: Atmosphere (A), Ocean (O), Terrestrial (T)
    A, O, T = y

    # Adjust fluxes for feedback mechanisms
    adjusted_k_ao = k_ao * (1 - feedback_strength * temperature)
    adjusted_k_at = k_at * (1 - feedback_strength * temperature)

    # Fluxes between reservoirs
    F_ao = adjusted_k_ao * A - k_oa * O
    F_at = adjusted_k_at * A - k_ta * T

    # Differential equations
    dA_dt = emissions_rate - F_ao - F_at
    dO_dt = F_ao
    dT_dt = F_at

    return np.array([dA_dt, dO_dt, dT_dt])

def radiative_forcing(atmosphere_co2):
    # Radiative forcing (W/m^2)
    return climate_sensitivity * np.log(atmosphere_co2 / preindustrial_co2)

def temperature_change(radiative_forcing):
    # Global temperature change (Celsius)
    return lambda_sensitivity * radiative_forcing

def heat_absorption(temperature):
    # Heat absorption using the Stefan-Boltzmann law (simplified)
    sigma = 5.67e-8  # Stefan-Boltzmann constant (W/m^2/K^4)
    earth_emissivity = 0.612  # Effective emissivity of Earth
    earth_temp_k = 288 + temperature  # Convert temperature change to Kelvin
    absorbed_heat = earth_emissivity * sigma * (earth_temp_k**4)
    return absorbed_heat

# Numerical integration using Euler's method
def integrate_euler(func, y0, t, dt, emissions_rate):
    y = np.zeros((len(t), len(y0)))
    y[0] = y0
    temperature = 0
    absorbed_heat = []

    for i in range(1, len(t)):
        forcing = radiative_forcing(y[i - 1][0])
        temperature = temperature_change(forcing)
        absorbed_heat.append(heat_absorption(temperature))
        y[i] = y[i - 1] + func(y[i - 1], emissions_rate, temperature) * dt

    return y, temperature, absorbed_heat

# Time span for simulation (years)
t_max = 400
dt = 0.1
time = np.arange(0, t_max + dt, dt)
initial_conditions = np.array([initial_atmosphere, initial_ocean, initial_terrestrial])

# Perform numerical integration
results, final_temperature, absorbed_heat = integrate_euler(co2_flux, initial_conditions, time, dt, emissions_rate)

# Extract results
atmosphere_co2 = results[:, 0]
ocean_co2 = results[:, 1]
terrestrial_co2 = results[:, 2]

# Calculate radiative forcing and temperature change over time
forcing = radiative_forcing(atmosphere_co2)
temperature = lambda_sensitivity * forcing

# Plot CO2 dynamics using scatter plots
plt.figure(figsize=(12, 12))

plt.subplot(3, 1, 1)
plt.scatter(time, atmosphere_co2, label='Atmosphere (ppm)', color='blue', s=10)
plt.scatter(time, ocean_co2, label='Oceans (GtC)', color='green', s=10)
plt.scatter(time, terrestrial_co2, label='Terrestrial (GtC)', color='brown', s=10)
plt.title('CO2 Reservoir Dynamics')
plt.xlabel('Time (years)')
plt.ylabel('CO2 Levels')
plt.legend()
plt.grid()

# Phase plane analysis for Atmosphere vs. Ocean
plt.subplot(3, 1, 2)
plt.plot(atmosphere_co2, ocean_co2, label='Phase Plane: Atmosphere vs Ocean', color='purple')
plt.title('Phase Plane Analysis')
plt.xlabel('Atmosphere CO2 (ppm)')
plt.ylabel('Ocean CO2 (GtC)')
plt.legend()
plt.grid()

# Plot absorbed heat over time
plt.subplot(3, 1, 3)
plt.plot(time[:-1], absorbed_heat, label='Absorbed Heat (W/m^2)', color='orange')
plt.title('Heat Absorption Over Time')
plt.xlabel('Time (years)')
plt.ylabel('Heat Absorbed (W/m^2)')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

# Bifurcation analysis with varying emissions rates
plt.figure(figsize=(8, 6))
emission_scenarios = [5, 10, 15, 20]
for emissions in emission_scenarios:
    res, _, _ = integrate_euler(co2_flux, initial_conditions, time, dt, emissions)
    plt.plot(time, res[:, 0], label=f'Emissions: {emissions} GtC/year')

plt.title('Bifurcation Analysis: Emissions Scenarios')
plt.xlabel('Time (years)')
plt.ylabel('Atmospheric CO2 (ppm)')
plt.legend()
plt.grid()
plt.show()

# Sensitivity analysis function
def sensitivity_analysis(param_name, param_values):
    results = []
    for value in param_values:
        globals()[param_name] = value
        res, final_temp, _ = integrate_euler(co2_flux, initial_conditions, time, dt, emissions_rate)
        atmosphere_co2 = res[:, 0]
        forcing = radiative_forcing(atmosphere_co2)
        temperature = temperature_change(forcing)
        results.append(temperature[-1])

    plt.bar(param_values, results, color='orange', label=f'Sensitivity: {param_name}')
    plt.title('Sensitivity Analysis')
    plt.xlabel(param_name)
    plt.ylabel('Final Temperature Change (°C)')
    plt.grid()
    plt.legend()
    plt.show()

# Example sensitivity analysis for climate sensitivity
sensitivity_analysis('climate_sensitivity', np.linspace(4, 6, 5))

# Stability analysis
plt.figure(figsize=(8, 6))
stability_conditions = [
    integrate_euler(co2_flux, [850, 38000, 2000], time, dt, emissions_rate * f)[0]
    for f in [0.5, 1, 1.5]
]
colors = ['blue', 'green', 'red']
labels = ['50% Emissions', '100% Emissions', '150% Emissions']

for i, cond in enumerate(stability_conditions):
    plt.plot(time, cond[:, 0], label=f'{labels[i]}', color=colors[i])

plt.title('Stability Analysis of Atmospheric CO2')
plt.xlabel('Time (years)')
plt.ylabel('Atmospheric CO2 (ppm)')
plt.legend()
plt.grid()
plt.show()

# Updated CO2 flux function with mitigation
def co2_flux_with_mitigation(y, emissions_rate, temperature, sequestration_rate=0):
    A, O, T = y

    # Adjust fluxes for feedback mechanisms
    adjusted_k_ao = k_ao * (1 - feedback_strength * temperature)
    adjusted_k_at = k_at * (1 - feedback_strength * temperature)

    # Fluxes between reservoirs
    F_ao = adjusted_k_ao * A - k_oa * O
    F_at = adjusted_k_at * A - k_ta * T

    # Differential equations including carbon sequestration
    dA_dt = emissions_rate - F_ao - F_at - sequestration_rate
    dO_dt = F_ao
    dT_dt = F_at

    return np.array([dA_dt, dO_dt, dT_dt])

# Scenario: Gradual reduction in emissions
def mitigation_scenario(time, y0, dt, initial_emissions, reduction_rate, sequestration_rate):
    emissions_rate = initial_emissions
    results = []
    y = y0
    temperature_trajectory = []
    absorbed_heat = []

    for t in time:
        forcing = radiative_forcing(y[0])
        temperature = temperature_change(forcing)
        absorbed_heat.append(heat_absorption(temperature))
        results.append(y)
        y = y + co2_flux_with_mitigation(y, emissions_rate, temperature, sequestration_rate) * dt

        # Gradual emissions reduction
        emissions_rate = max(0, emissions_rate - reduction_rate * dt)

    results = np.array(results)
    return results, temperature_trajectory, absorbed_heat

# Time span and mitigation parameters
time = np.arange(0, t_max + dt, dt)
initial_emissions = 10  # Initial emissions rate in GtC/year
reduction_rate = 0.05  # Reduction rate of emissions per year
sequestration_rate = 2  # Carbon sequestration rate in GtC/year

# Simulate mitigation scenarios
results_mitigation, _, absorbed_heat_mitigation = mitigation_scenario(
    time, initial_conditions, dt, initial_emissions, reduction_rate, sequestration_rate
)

# Extract results for plotting
atmosphere_co2_mitigation = results_mitigation[:, 0]

# Plot CO2 trajectories
plt.figure(figsize=(10, 6))
plt.plot(time, atmosphere_co2, label='Baseline Emissions (No Mitigation)', color='blue')
plt.plot(time, atmosphere_co2_mitigation, label='Mitigation Scenario (Reduction + Sequestration)', color='green')
plt.title('Atmospheric CO2 with Mitigation Strategies')
plt.xlabel('Time (years)')
plt.ylabel('Atmospheric CO2 (ppm)')
plt.legend()
plt.grid()
plt.show()

# Plot temperature trajectories
temperature_baseline = lambda_sensitivity * radiative_forcing(atmosphere_co2)
temperature_mitigation = lambda_sensitivity * radiative_forcing(atmosphere_co2_mitigation)

plt.figure(figsize=(10, 6))
plt.plot(time, temperature_baseline, label='Baseline Emissions (No Mitigation)', color='blue')
plt.plot(time, temperature_mitigation, label='Mitigation Scenario', color='green')
plt.title('Temperature Change with Mitigation Strategies')
plt.xlabel('Time (years)')
plt.ylabel('Temperature Change (°C)')
plt.legend()
plt.grid()
plt.show()



