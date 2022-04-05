import pandas as pd
from tabulate import tabulate
import dill
from sklearn.model_selection import train_test_split
from config import Config
#from app.model.datapipeline import DataPipeline
from datapipeline import DataPipeline

df = pd.read_csv(f"{Config.source_dir}master.csv")
df.columns = df.columns.str.replace('($)', '', regex=False)
df.columns = df.columns.str.strip()

print(tabulate(df.head(5), headers='keys', tablefmt='fancy_grid'))

f = open(f"{Config.source_dir}master_info.txt", 'w+')
df.info(buf=f)
f.close()

X_train, X_test, y_train, y_test = train_test_split(df, df['suicides_no'], test_size=0.33, random_state=42)
# save test
X_test.to_csv(f'{Config.train_test_dir}X_test.csv', index=None)
y_test.to_csv(f'{Config.train_test_dir}y_test.csv', index=None)
# save train
X_train.to_csv(f'{Config.train_test_dir}X_train.csv', index=None)
y_train.to_csv(f'{Config.train_test_dir}y_train.csv', index=None)

pipeline = DataPipeline()
pipeline.fit(X_train, y_train)

with open(f'{Config.dumps_dir}pipeline.dill', "wb") as f:
    dill.dump(pipeline, f)
