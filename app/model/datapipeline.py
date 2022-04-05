from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np


class DataPipeline:

    def __init__(self):
        self.cat_features = ['country']
        self.dummy_features = ['age', 'generation']
        self.bin_features = ['sex']
        self.str_numbers_features = ['gdp_for_year']
        self.num_features = ['year', 'population', 'gdp_per_capita']
        self.target = 'suicides_no'

        self.__pipeline = None

        self.__pipeline_init()

    def fit(self, X, y):
        return self.__pipeline.fit(X, y)

    def predict(self, X):
        preds = pd.DataFrame(self.__pipeline.predict(X), columns=[self.target])
        preds.round()
        preds = preds.astype(int)
        return preds

    class FeatureSelector(BaseEstimator, TransformerMixin):
        def __init__(self, column):
            self.column = column

        def fit(self, X, y=None):
            return self

        def transform(self, X, y=None):
            return X[self.column]

    class NumberSelector(BaseEstimator, TransformerMixin):
        """
        Transformer to select a single column from the data frame to perform additional transformations on
        Use on numeric columns in the data
        """

        def __init__(self, key):
            self.key = key

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            return X[[self.key]]

    class OHEEncoder(BaseEstimator, TransformerMixin):
        def __init__(self, key):
            self.key = key
            self.columns = []

        def fit(self, X, y=None):
            self.columns = [col for col in pd.get_dummies(X, prefix=self.key).columns]
            return self

        def transform(self, X):
            X = pd.get_dummies(X, prefix=self.key)
            test_columns = [col for col in X.columns]
            for col_ in self.columns:
                if col_ not in test_columns:
                    X[col_] = 0
            return X[self.columns]

    class OHEEncoderBin(BaseEstimator, TransformerMixin):
        def __init__(self, key):
            self.key = key
            self.columns = []

        def fit(self, X, y=None):
            B = [col for col in pd.get_dummies(X, prefix=self.key).columns]
            B.sort()
            self.columns = B[:1]
            return self

        def transform(self, X):
            X = pd.DataFrame(pd.get_dummies(X, prefix=self.key).iloc[0].values, columns=self.columns)
            return X[self.columns]

    class StrToNumber(BaseEstimator, TransformerMixin):
        def __init__(self, key):
            self.key = key
            self.columns = []

        def fit(self, X, y=None):
            self.columns = [self.key]
            return self

        def transform(self, X):
            X = X.apply(lambda x: x.replace(r'\D+', '', regex=True)).astype(np.int64)
            return X[self.columns]

    class CatTransformer(BaseEstimator, TransformerMixin):
        def __init__(self, key):
            self.key = key

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            X[self.key] = X[self.key]
            return X

    def __pipeline_init(self):
        final_transformers = list()

        for cat_col in self.dummy_features:
            cat_transformer = Pipeline([
                ('selector', self.FeatureSelector(column=cat_col)),
                ('ohe', self.OHEEncoder(key=cat_col))
            ])
            final_transformers.append((cat_col, cat_transformer))

        for cont_col in self.num_features:
            cont_transformer = Pipeline([
                ('selector', self.NumberSelector(key=cont_col)),
                ('Scale', StandardScaler())
            ])
            final_transformers.append((cont_col, cont_transformer))

        for bin_col in self.bin_features:
            bin_transformer = Pipeline([
                ('selector', self.FeatureSelector(column=bin_col)),
                #('ohe', self.OHEEncoderBin(key=bin_col))
                ('ohe', self.OHEEncoder(key=bin_col))
            ])
            final_transformers.append((bin_col, bin_transformer))

        for str_num_col in self.str_numbers_features:
            cont_transformer = Pipeline([
                ('selector', self.NumberSelector(key=str_num_col)),
                ('to_number', self.StrToNumber(key=str_num_col)),
                ('Scale', StandardScaler())
            ])
            final_transformers.append((str_num_col, cont_transformer))

        # for cat_col in self.cat_features:
        #     cat_transformer = Pipeline([
        #         ('pass', self.CatTransformer(cat_col)),
        #         ('selector', self.FeatureSelector(column=cat_col))
        #
        #     ])
        #     final_transformers.append((cat_col, cat_transformer))

        feats = FeatureUnion(final_transformers)

        self.__pipeline = Pipeline([
            ('features', feats),
            #('regressor', LogisticRegression(random_state=42, max_iter=1000)),
            ('regressor', RandomForestRegressor(random_state=42)),
        ])
