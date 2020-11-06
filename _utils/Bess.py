# battery mode (precharge (up to 10% capacity), Current mode (10 to 80%), CV (80 to 100%))
# settings (capacity, max current)
# charge algorithm (constant current, variable current, etc)

# BATTERY MODELS
# https://www.mathworks.com/help/physmod/sps/powersys/ref/battery.html#bry4req-2
# https://github.com/susantoj/kinetic-battery/blob/master/kinetic_battery.py
# https://www.homerenergy.com/products/pro/docs/3.11/creating_an_idealized_storage_component.html

# TODO: revamp status to give the SoC at the beginning and end of the current time step
# TODO: simplify scheduling to the net energy activity of the battery (assuming constant power, within max bounds)
# TODO: add functions to check schedule
# TODO: at some point make sure all user submitted time intervals are evenly divisible by round duration

class Storage:
    """This is a stripped down battery simulator for best response
    It approximates the behaviour of the BESS module in TREX-Core

    # positive charge = charge
    # negative charge = discharge

    # efficiency is one way efficiency. Round trip efficiency is efficiecy^2
    # standard units used are:
    # power: W; Energy: Wh; time: s
    # self discharge is assumed to be a constant.

    # charge and discharge are measured externally at the meter, which means that 
    # internally it will charge less than metered and discharge more than metered due to non-ideal efficiency

    """

    def __init__(self, capacity=7000, power=3300, efficiency=0.95):
        self.__info = {
            'capacity': capacity,  # Wh
            'efficiency': efficiency,  # charges less than meter readings, and discharges more than meter readings
            'power_rating': power,  # reading at the meter in W
        }

    # Collect status of battery charge at start and end of turn/ current scheduled battery charge or discharge
    def simulate_activity(self, start_energy, energy_activity=0):
        # Bess_output_kW, Soc_now = battery_function(desired_bess_output, Soc_before)
        projected_energy = max(0, min(self.__info['capacity'], start_energy + energy_activity))
        charge_cap = (self.__info['capacity'] - projected_energy) / self.__info['efficiency']  # at the meter
        discharge_cap = projected_energy * self.__info['efficiency']  # at the meter
        actual_energy_activity = 0
        if energy_activity > 0:
            actual_energy_activity = min(charge_cap, energy_activity)
        else:
            actual_energy_activity = max(discharge_cap, energy_activity)

        end_energy = start_energy + energy_activity
        return actual_energy_activity, end_energy

    def get_info(self):
        return self.__info

    def reset(self, soc_pct):
        state_of_charge = int(self.__info['capacity'] * max(0, min(100, soc_pct/100)))
        self.__info['state_of_charge'] = state_of_charge