import numpy as np
import matplotlib.pyplot as plt
import dataset
from _utils import utils

db_path = "postgresql://postgres:postgres@localhost/SunDance"
profile_name = "SunDance_10011"
db = dataset.connect(db_path)
profile_table = db[profile_name]

start_datetime = "2015-06-01 0:0:0"
end_datetime = "2015-06-2 0:0:0"
timezone = "America/Vancouver"

t_start = utils.timestr_to_timestamp(start_datetime, timezone)
t_end = utils.timestr_to_timestamp(end_datetime, timezone)
result = profile_table.find(time={'between': [t_start, t_end]})

time = list()
consumption = list()
generation = list()

row_count = 0
for row in result:
    # time.append(row['time'])
    time.append(row_count)
    consumption.append(row['consumption'])
    generation.append(row['generation'])
    row_count += 1

# plot
fig, ax = plt.subplots()
ax.plot(time, generation, label="generation")
ax.plot(time, consumption, label="consumption")
# ax.set(xlim=(0, 23), xticks=np.arange(0, 24))
plt.legend()
plt.grid()
plt.show()



