import numpy as np

import xgboost as xgb

from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


class MyXGBClassifier(object):

    def __init__(self, booster_params, fit_params, plot_curves=False):
        self.booster_params = booster_params.copy()
        self.fit_params = fit_params.copy()
        self.plot_curves = plot_curves
        if isinstance(self.fit_params.get('learning_rates', None), (float, int)):
            self.fit_params['learning_rates'] = list(self.fit_params['learning_rates'] * \
                                               np.ones(self.fit_params.get('num_boost_round', 10)).astype(float))
        self.le = LabelEncoder()

    def fit(self, X, y, CV=None):
        y_e = self.le.fit_transform(y)
        self.booster_params['num_class'] = self.le.classes_.shape[0]

        if isinstance(self.booster_params.get('colsample_bytree', None), int):
            self.booster_params['colsample_bytree'] = self.booster_params['colsample_bytree'] / X.shape[1]
            #print(self.booster_params['colsample_bytree'])
        if isinstance(self.booster_params.get('colsample_bylevel', None), int):
            self.booster_params['colsample_bylevel'] = self.booster_params['colsample_bylevel'] / \
                                                       X.shape[1] / \
                                                       self.booster_params['colsample_bytree']
            #print(self.booster_params['colsample_bylevel'])


        X = xgb.DMatrix(X, label=y_e)

        if CV is not None and isinstance(CV, (list, tuple, np.ndarray)) and len(CV) ==2:
            X_cv = xgb.DMatrix(CV[0], label=self.le.transform(CV[1]))
            self.fit_params['evals'] = [(X, 'train'), (X_cv, 'eval'),]
            self.evals_result = {}
            self.fit_params['evals_result'] = self.evals_result


        self.bst = xgb.train(params=self.booster_params, dtrain=X, **self.fit_params)

        if self.plot_curves:
            self._plot_curves()

    def predict(self, X):
        X = xgb.DMatrix(X)
        return self.bst.predict(X)

    def predict_proba(self, X):
        return self.predict(X)


    def _plot_curves(self):

        fig = plt.figure(figsize=(5, 5))
        _curves = {}
        y_min = 1e111
        y_max = -1e111
        for k, v in self.evals_result.items():
            for m, res in v.items():
                _curves["{}, {}".format(k, m)] = res
                y_min = min(y_min, min(res))
                y_max = max(y_max, np.percentile(res, 80))

        for lab, res in _curves.items():
            plt.loglog(res, label=lab)

        y_min = 0.99 * y_min
        y_max = 1.01 * y_max

        fig.axes[0].set_ylim(y_min, y_max)
        fig.axes[0].yaxis.set_ticks(np.linspace(y_min, y_max, 11))
        fig.axes[0].yaxis.set_ticklabels(np.linspace(y_min, y_max, 11))

        plt.legend(loc=0)

