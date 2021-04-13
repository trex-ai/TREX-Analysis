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

    def __init__(self, capacity=7000, power=3300):
        # assume perfect efficiency
        self.__info = {
            'capacity': capacity,  # Wh
            # 'efficiency': efficiency,  # charges less than meter readings, and discharges more than meter readings
            'power_rating': power,  # reading at the meter in W
        }

    # Collect status of battery charge at start and end of turn/ current scheduled battery charge or discharge
    def simulate_activity(self, start_energy:int, target_energy=0, duration_s=60):
        storage_capacity = self.__info['capacity']
        start_energy = max(0, min(start_energy, storage_capacity))
        duration_h = duration_s / 3600
        target_power = target_energy / duration_h

        if target_energy > 0:
            actual_power = min(target_power, self.__info['power_rating'])
            actual_energy_max = int(actual_power * duration_h)
            actual_energy_capped = min(actual_energy_max, storage_capacity - start_energy)
            end_energy = start_energy + actual_energy_capped
            return actual_energy_capped, end_energy
        elif target_energy < 0:
            actual_power = max(target_power, -self.__info['power_rating'])
            actual_energy_max = int(actual_power * duration_h)
            actual_energy_capped = max(-start_energy, actual_energy_max)
            end_energy = start_energy + actual_energy_capped
            return actual_energy_capped, end_energy
        else:
            return 0, start_energy

    def get_info(self):
        return self.__info