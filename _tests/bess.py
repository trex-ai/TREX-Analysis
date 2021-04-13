from _utils import Bess
battery = Bess.Storage(capacity=7000, power=3300)

print('normal charge:', battery.simulate_activity(0, 10, 60))
print('overcharge 1:', battery.simulate_activity(0, 7000, 60))
print('overcharge 2:', battery.simulate_activity(7000, 10, 60))
print('normal discharge:', battery.simulate_activity(7000, -10, 60))
print('over-discharge 1:', battery.simulate_activity(7000, -7000, 60))
print('over-discharge 2:', battery.simulate_activity(0, -10, 60))