from sqlalchemy import create_engine
import dataset
import pandas as pd

class extractor():
    def __init__(self, db_path):
        self.database = db_path

    def from_metrics(self,db_path, gen, table_name, agent_id, **kwargs):
        table_name = '_'.join([str(gen), sim_type, 'metrics', agent_id])

        db = dataset.connect(db_path)
        table = db[table_name]

        data = []
        for row in table:
            data.append(row)

        if data:
            dicter = {k: [dic[k] for dic in data] for k in data[0]}
        else:
            dicter = {}

        return pd.DataFrame.from_dict(dicter)

    def from_transactions(self):

        return False
