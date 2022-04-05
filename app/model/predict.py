import pandas as pd
from sklearn.metrics import r2_score, median_absolute_error, mean_squared_error
import dill
from config import Config


X_test = pd.read_csv(f'{Config.train_test_dir}X_test.csv')
y_test = pd.read_csv(f'{Config.train_test_dir}y_test.csv')

with open(f'{Config.dumps_dir}pipeline.dill', 'rb') as in_strm:
    pipeline = dill.load(in_strm)

preds = pipeline.predict(X_test)

preds.to_csv(f'{Config.prediction_dir}test_predictions.csv', index=None)

r2 = r2_score(y_test, preds)
mae = median_absolute_error(y_test, preds)
mse = mean_squared_error(y_test, preds)

with open(f'{Config.prediction_dir}metrics.txt', "w") as f:
    f.write(f'r2_score={r2}, mae={mae:.3f}, mse={mse:.3f}')
    f.close()
