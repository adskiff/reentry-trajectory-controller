import math
import numpy as np
import csv
from scipy.integrate import solve_ivp
from datetime import datetime, date, time, timezone
from pyatmos import coesa76
current_time = datetime.now(timezone.utc)


## take starting orbital parameters

def get_flight_data():
    latitude = float(input('Starting latitude in degrees: '))
    longitude = float(input('Starting longitude in degrees: '))
    initial_alt = float(input('Starting altitude in meters: '))
    initial_vel = float(input('Starting orbital velocity in meters per second: '))
    initial_gamma = float(input('Starting attitude in degrees (below horizion is positive): '))
    ref_area = float(input('Windward side reference area parallel to velocity (circular shape assumed): '))
    vehicle_mass = float(input('Mass of vessel: '))
    RB = float(input('Maximum transverse radius: '))
    RN = float(input('Heat shield radius: '))
    return [latitude, longitude, initial_alt, initial_vel, initial_gamma, ref_area, vehicle_mass, RB, RN]



# Call initial parameters
init_params = get_flight_data()
INIT_GAMMA_RADIANS = init_params[4] * (math.pi / 180)
EARTH_ECCENTRICITY = 0.0818
RADIAN_LAT = init_params[0] * (math.pi / 180)
EARTH_EQUITORIAL_RADIUS = 6378100
SEA_LEVEL_AIR_PRESSURE = 101325
EARTH_RADIUS_AT_LATITUDE = EARTH_EQUITORIAL_RADIUS * math.sqrt((1 - ((2 * (EARTH_ECCENTRICITY ** 2 )) * (EARTH_ECCENTRICITY ** 4)) * (math.sin(RADIAN_LAT) ** 2))
                       / (1 - ((EARTH_ECCENTRICITY ** 2) * (math.sin(RADIAN_LAT) ** 2))))
EARTH_SURF_GRAV = 9.807
INIT_ATMOS = coesa76(init_params[2] / 1000)
INIT_DYN_PRESS = (1/2) * INIT_ATMOS.rho * init_params[3] # velocity



# constants dictionary

constants = {
    'g': EARTH_SURF_GRAV,
    'initial_air_pressure': INIT_DYN_PRESS,
    'SEA_LEVEL_AIR_PRESSURE': SEA_LEVEL_AIR_PRESSURE,
    'RB': init_params[7],
    'RN': init_params[8],
    'vehicle_mass': init_params[6],
    'ref_area': init_params[5],
    'gamma': INIT_GAMMA_RADIANS,
    'latitude_radius' : EARTH_RADIUS_AT_LATITUDE
}

## equations of motion

def density_ratio(gamma):
    return (gamma - 1) / (gamma + 1)

def phi(RB, RN):
    return math.asin(RB / RN)

def zero_lift_drag_coefficient(epsilon, phi):
    return (1 - (epsilon / 2)) * (1 + (math.cos(phi) ** 2))

def lift_coefficient(Cd0, alpha, epsilon, phi):
    return (Cd0 * math.cos(alpha) ** 2 + (1/2 * (1 - (epsilon/2)) * math.sin(phi) ** 2 * (3 * (math.sin(alpha) ** 2) - 2)) * math.sin(alpha))

def drag_coefficient(Cd0, alpha, epsilon, phi):
    return (Cd0 * (math.cos(alpha) ** 2) + (3/2 * (1 - (epsilon/2)) * math.sin(phi) ** 2 * (math.sin(alpha) ** 2)) * math.cos(alpha))

def dynamic_pressure(air_pressure, airspeed):
    return 0.5 * air_pressure * (airspeed ** 2)

def lift(q, Cl, ref_area):
    return q * (Cl * ref_area)

def drag(q, Cd, ref_area):
    return q * (Cd * ref_area)


def gravity(EARTH_SURF_GRAV, EARTH_RADIUS_AT_LATITUDE):
    return EARTH_SURF_GRAV * (6310000 / (EARTH_RADIUS_AT_LATITUDE))

def ballistic_coefficient(vehicle_mass, Cd, ref_area):
    return vehicle_mass / (Cd * ref_area)

def equations_of_motion(state, constants_dict, alpha):
    t, v, gamma, altitude = state

    g = constants_dict['g']
    SEA_LEVEL_AIR_PRESSURE = constants_dict['SEA_LEVEL_AIR_PRESSURE']
    RB = constants_dict['RB']
    RN = constants_dict['RN']
    vehicle_mass = constants_dict['vehicle_mass']
    ref_area = constants_dict['ref_area']
    rad_lat = constants_dict['latitude_radius']

    epsilon = density_ratio(gamma)

    Cd0 = zero_lift_drag_coefficient(epsilon, phi(RB, RN))
    Cl = lift_coefficient(Cd0, alpha, epsilon, phi(RB, RN))
    Cd = drag_coefficient(Cd0, alpha, epsilon, phi(RB, RN))
    B = ballistic_coefficient(vehicle_mass, Cd, ref_area)

    atmos = coesa76(altitude / 1000)
    air_density = atmos.rho
    q = dynamic_pressure(air_density, v)
    L = lift(q, Cl, ref_area)
    D = drag(q, Cd, ref_area)

    dt = 1
    dv = (-(D / (vehicle_mass * g)) - math.sin(gamma)) * g
    dgamma = ((((vehicle_mass * (v ** 2)) / (rad_lat + altitude) ) * math.cos(gamma)) + L - (vehicle_mass * g * math.cos(gamma))) / (vehicle_mass * v)
    daltitude = v * math.sin(gamma)

    return [dt, dv, dgamma, daltitude]

def modified_equations_of_motion(t, state, constants_dict):
    alpha = g_force_controller(t, state, constants_dict)
    return equations_of_motion(state, constants_dict, alpha)

def g_force_controller(t, state, constants_dict):
    # Implement a simple proportional controller to maintain g-force within limits
    dt, dv, _, _ = equations_of_motion(state, constants_dict, 0)
    current_g_force = abs(dv) / dt
    target_g_force = 1.5  # Change this value to set the desired g-force
    Kp = 0.1  # Proportional gain

    error = target_g_force - current_g_force
    alpha = Kp * error

    return alpha

def run_simulation(initial_state, constants_dict, t_start, t_end):
    result = solve_ivp(
        modified_equations_of_motion,
        (t_start, t_end),
        initial_state,
        args=(constants_dict,),
        dense_output=True,
        rtol=1e-6,
        atol=1e-9,
    )

    return result

def main():
    # Initial state
    initial_state = [0, init_params[3], INIT_GAMMA_RADIANS, init_params[2]]

    # Set the start and end time for the simulation
    t_start = 0
    t_end = 10000  # Increase this value if needed

    result = run_simulation(initial_state, constants, t_start, t_end)

    # Find the time when the altitude is less than or equal to 5,000 meters
    termination_altitude = 5000
    termination_index = np.where(result.y[3] <= termination_altitude)[0][0]
    termination_time = result.t[termination_index]

    print(f"Termination time: {termination_time:.2f} seconds")
    print(f"Termination altitude: {result.y[3][termination_index]:.2f} meters")

if __name__ == "__main__":
    main()