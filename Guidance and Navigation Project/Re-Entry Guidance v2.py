import math
import numpy as np
import csv
import os
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
    initial_gamma = float(input('Starting attitude in degrees (below horizion is negative): '))
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

def equations_of_motion(t, state, constants_dict, alpha):
    _, v, gamma, altitude = state


    g = constants_dict['g']
    RB = constants_dict['RB']
    RN = constants_dict['RN']
    vehicle_mass = constants_dict['vehicle_mass']
    ref_area = constants_dict['ref_area']
    rad_lat = constants_dict['latitude_radius']

    epsilon = density_ratio(gamma)

    Cd0 = zero_lift_drag_coefficient(epsilon, phi(RB, RN))
    Cl = lift_coefficient(Cd0, alpha, epsilon, phi(RB, RN))
    Cd = drag_coefficient(Cd0, alpha, epsilon, phi(RB, RN))

    atmos = coesa76(altitude / 1000)
    air_density = atmos.rho
    q = dynamic_pressure(air_density, v)
    L = lift(q, Cl, ref_area)
    D = drag(q, Cd, ref_area)

    dt = 1
    dv = (-(D / (vehicle_mass * g)) - math.sin(gamma)) * g
    dgamma = ((((vehicle_mass * (v ** 2)) / (rad_lat + altitude)) * math.cos(gamma)) + L - (vehicle_mass * g * math.cos(gamma))) / (vehicle_mass * v)
    daltitude = v * math.sin(gamma)

    return np.array([dv, dgamma, daltitude, 0], dtype=object)

def run_simulation_rk4(initial_state, constants_dict, alpha_controller, pid_controller, time_limit=10000):
    def wrapped_equations_of_motion(t, state):
        alpha = alpha_controller(t, state, constants_dict, pid_controller)
        return equations_of_motion(t, state, constants_dict, alpha)

    t_eval = np.arange(0, time_limit + 1, 1)
    sol = solve_ivp(wrapped_equations_of_motion, (0, time_limit), initial_state, t_eval=t_eval, method='RK45')

    states_over_time = sol.y.T
    times_over_time = sol.t

    return states_over_time, times_over_time

class PIDController:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.previous_error = 0
        self.integral = 0

    def update(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.previous_error) / dt
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.previous_error = error
        return output

def g_force_controller_pid(t, state, constants_dict, pid_controller):
    _, v, gamma, altitude = state
    _, dv, _, _ = equations_of_motion(t, state, constants_dict, 0)

    current_g_force = abs((v ** 2 / (constants_dict['latitude_radius'] + altitude)) + constants_dict['g'] * math.cos(gamma) - dv) / constants_dict['g']
    target_g_force = 5  # Change this value to set the desired g-force

    error = target_g_force - current_g_force
    alpha = pid_controller.update(error, 1)

    return alpha

def run_simulation(initial_state, constants_dict, alpha_controller, time_limit=10000):
    state = initial_state
    states_over_time = [initial_state]
    times_over_time = [0]

    for i in range(time_limit):
        alpha = alpha_controller(state, constants_dict)
        next_state = equations_of_motion(state, constants_dict, alpha)
        state = [state[j] + next_state[j] for j in range(len(state))]
        states_over_time.append(state)
        times_over_time.append(i + 1)

        if state[3] <= 5000:
            if state[1] <= 120:
                print("Safe to deploy parachutes!")
            else:
                print("Unsafe velocity!")
            break

        if state[3] < 0:
            break

    return states_over_time, times_over_time


def save_to_csv(data, extra_data, timestamps, filename='simulation_results.csv'):
    headers = ['Time (s)', 'Velocity (m/s)', 'Gamma (rad)', 'Altitude (m)', 'G-force', 'Angle of Attack (rad)', 'AoA Adjustment (rad)']
    
    with open(filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(headers)

        for t, row, extra_row in zip(timestamps, data, extra_data):
            csv_writer.writerow([t] + row + extra_row)

def main():
    # Call initial parameters
    initial_state = [0, init_params[3], init_params[4] * (math.pi / 180), init_params[2]]

    pid_controller = PIDController(Kp=0.1, Ki=0.05, Kd=0.01)

    # Run the simulation
    states_over_time, times_over_time = run_simulation_rk4(initial_state, constants, g_force_controller_pid, pid_controller)

    # Calculate g-force, angle of attack, and AoA adjustment for each time step
    extra_data = []
    for state in states_over_time:
        alpha = g_force_controller_pid(state, constants)
        _, v, gamma, altitude = state
        _, dv, _, _ = equations_of_motion(state, constants, alpha)
        current_g_force = (abs((v ** 2 / (constants['latitude_radius'] + altitude)) + constants['g'] * math.cos(gamma) - dv) / constants['g'])
        aoa_adjustment = alpha * 0.1
        extra_data.append([current_g_force, alpha, aoa_adjustment])

    # Save the results to a CSV file
    save_to_csv(states_over_time, extra_data, times_over_time)

if __name__ == "__main__":
    main()