from sqlalchemy import and_, or_
import dataset
import pandas as pd
import json

class Extractor():
    def __init__(self, db_path):
        self.database = db_path

    def from_metrics(self, gen, sim_type, agent_id, **kwargs):
        table_name = '_'.join([str(gen), sim_type, 'metrics', agent_id])

        db = dataset.connect(self.database)
        table = db[table_name]

        data = []
        for row in table:
            data.append(row)

        if data:
            dicter = {k: [dic[k] for dic in data] for k in data[0]}
        else:
            dicter = {}

        return pd.DataFrame.from_dict(dicter)

    def from_transactions(self, db_path, table_name, time_consumption_interval:tuple):
        db = dataset.connect(db_path)
        table = db[table_name].table
        statement = table.select().where(and_(table.c.time_consumption >= time_consumption_interval[0],
                                              table.c.time_consumption <= time_consumption_interval[1]))
        transactions = db.query(statement)

        return pd.DataFrame(transactions)

    def extract_config(self):
        table = 'configs'
        db = dataset.connect(self.database)
        tab = db[table]
        data = []
        for row in tab:
            data.append(row)

        return json.dump(data[0])

agent_id = 'egauge19821'
db_path1 = 'postgresql://postgres:postgres@stargate/remote_agent_test_np'
sim_type = 'training'
extractor = Extractor(db_path1)
dataframe = extractor.from_metrics(0, sim_type, agent_id)

print(dataframe.head())