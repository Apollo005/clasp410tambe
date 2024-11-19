import numpy as np
import matplotlib.pyplot as plt

# Constants from Table 1
S0 = 1370  # Solar flux, W/m^2
C = 4.2e6  # Heat capacity of seawater, J/(m^3 K)
Rho = 1020  # Density of seawater, kg/m^3
Lambda = 100  # Thermal diffusivity, m^2/s
Dz = 50  # Mixed-layer depth of ocean, m
Stefan_Boltzmann = 5.67e-8  # Stefan-Boltzmann constant, W/(m^2 K^4)

# Temperature thresholds for albedo change
Freeze_Temp = -10
Albedo_Water = 0.3
Albedo_Ice = 0.6

# Helper function for warm Earth temperature profile
def temp_warm(lats_in):
    T_warm = np.array([-47, -19, -11, 1, 9, 14, 19, 23, 25, 25, 23, 19, 14, 9, 1, -11, -19, -47])
    npoints = T_warm.size
    dlat = 180 / npoints
    lats = np.linspace(dlat / 2.0, 180 - dlat / 2.0, npoints)
    coeffs = np.polyfit(lats, T_warm, 2)
    temp = coeffs[2] + coeffs[1] * lats_in + coeffs[0] * lats_in ** 2
    return temp

# Helper function for insolation
def insolation(S0, lats):
    max_tilt = 23.5
    dlong = 0.01
    angle = np.cos(np.pi / 180.0 * np.arange(0, 360, dlong))
    angle[angle < 0] = 0
    total_solar = S0 * angle.sum()
    S0_avg = total_solar / (360 / dlong)

    insolation = np.zeros(lats.size)
    tilt = [max_tilt * np.cos(2.0 * np.pi * day / 365) for day in range(365)]

    for i, lat in enumerate(lats):
        zen = lat - 90.0 + tilt
        zen = np.clip(zen, -90, 90)
        insolation[i] = np.sum(np.cos(np.pi / 180.0 * zen))

    insolation = S0_avg * insolation / 365
    return insolation

# Numerical solver for temperature evolution
def solve_temperature(npoints, dt, years, initial_temp, gamma=1.0):
    dlat = 180 / npoints
    lats = np.linspace(dlat / 2.0, 180 - dlat / 2.0, npoints)

    albedo = np.full(npoints, Albedo_Water)
    T = np.array(initial_temp)

    for _ in range(int(years / dt)):
        S = insolation(S0 * gamma, lats) * (1 - albedo)
        radiative = S - Stefan_Boltzmann * T ** 4
        diffusion = np.zeros_like(T)

        for i in range(1, npoints - 1):
            diffusion[i] = Lambda * (T[i + 1] - 2 * T[i] + T[i - 1]) / (dlat ** 2)

        T += dt * (radiative / (Rho * C * Dz) + diffusion)

        loc_ice = T <= Freeze_Temp
        albedo[loc_ice] = Albedo_Ice
        albedo[~loc_ice] = Albedo_Water

    return T, lats

# Main function to explore scenarios
def main():
    npoints = 100
    dt = 1  # in years
    years = 10000

    # Initial conditions: warm Earth
    initial_temp = temp_warm(np.linspace(0, 180, npoints))

    # Solve for warm Earth equilibrium
    T_warm, lats = solve_temperature(npoints, dt, years, initial_temp)

    plt.plot(lats, T_warm, label="Warm Earth")

    # Initial conditions: cold Earth
    initial_temp = np.full(npoints, -60)
    T_cold, _ = solve_temperature(npoints, dt, years, initial_temp)

    plt.plot(lats, T_cold, label="Cold Earth")

    # Initial conditions: flash freeze
    initial_temp = temp_warm(np.linspace(0, 180, npoints))
    albedo = np.full(npoints, Albedo_Ice)
    T_flash, _ = solve_temperature(npoints, dt, years, initial_temp)

    plt.plot(lats, T_flash, label="Flash Freeze")

    plt.xlabel("Latitude (degrees)")
    plt.ylabel("Temperature (C)")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
