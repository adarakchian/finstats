import pandas as pd
import numpy as np
from statsmodels import api as sm
import typing
from sklearn import linear_model as lm


class RegressionResults:

    def __init__(self,
                 factor_performance: typing.Union[pd.DataFrame, pd.Series],
                 stock_performance: typing.Union[pd.Series],
                 regression: typing.Union[lm.LassoCV, lm.LinearRegression]
                 ):
        self.model = regression
        self.factor_performance = factor_performance
        self.stock_performance = stock_performance

        self.exposures = pd.Series(regression.coef_, index=self.factor_performance.columns).sort_values(ascending=False)
        self.fitted_values: pd.Series = regression.predict(self.factor_performance)
        self.residuals: pd.Series = stock_performance - self.fitted_values


class OuParameters:

    def __init__(self, residuals: np.ndarray):
        res = sm.tsa.ARIMA(residuals.cumsum(), order=(1, 0, 0)).fit()
        a, b = res.params
        var = res.resid.var()

        k = -np.log(b) * 252
        m = a / (1 - b)
        sigma = np.sqrt((var * 2 * k) / (1 - b ** 2))
        sigma_eq = np.sqrt(var / (1 - b ** 2))

        self.a: int = a
        self.b: int = b
        self.var: int = var
        self.k: int = k
        self.m: int = m
        self.sigma: int = sigma
        self.sigma_eq: int = sigma_eq
        self.tau: int = 1 / k


class OuProcessFitter:
    regression: RegressionResults
    ou_parameters: OuParameters

    def __init__(self,
                 factors: typing.Union[pd.DataFrame, pd.Series],
                 stock: pd.Series,
                 etf_map: typing.Dict[str, str]
                 ):
        self.factors = factors
        self.stock = stock
        self.stock_name = stock.name

        if factors.shape[0] != stock.shape[0]:
            raise IndexError("Mismatching series length")
        self.factor_performance = np.log(self.factors).diff().iloc[1:]
        self.stock_performance = np.log(self.stock).diff().iloc[1:]
        self.etf = self.factor_performance[[etf_map[str(stock.name)]]]

    def estimate_lasso(self):
        lasso_search = lm.LassoCV(cv=3, fit_intercept=True)
        search_result = lasso_search.fit(self.factor_performance, self.stock_performance)
        self.regression = RegressionResults(self.factor_performance, self.stock_performance, search_result)

    def estimate_industry_beta(self):
        reg = lm.LinearRegression()
        reg.fit(self.etf, self.stock_performance)
        self.regression = RegressionResults(self.etf, self.stock_performance, reg)

    def fit_ou(self):
        self.ou_parameters = OuParameters(self.regression.residuals.values)


class RelativeValueResults:
    s_score: int

    def __init__(self, ou_process: OuProcessFitter):
        self.ou_process: OuProcessFitter = ou_process

    def set_s_score(self, score):
        self.s_score = score


class RelativeValueEstimator:
    # fitted_values: typing.List[OuProcessFitter]
    fitted_values: typing.Dict[str, RelativeValueResults]
    _mean_m: int
    _excluded_stocks: typing.List[str]

    def __init__(self,
                 prices: pd.DataFrame,
                 etf_tickers: typing.List[str],
                 stock_tickers: typing.List[str],
                 stock_to_etf_map: typing.Dict[str, str]
                 ):
        self.prices = prices
        self.etf_tickers = etf_tickers
        self.stock_tickers = stock_tickers
        self.stock_to_etf_map = stock_to_etf_map

    def fit_ou_process(self):
        fitted_results = {}
        total_length = len(self.stock_tickers)
        excluded = []
        m_parameters = []
        for i, col in enumerate(self.stock_tickers):
            try:
                ou = OuProcessFitter(self.prices.loc[:, self.etf_tickers], self.prices.loc[:, col],
                                     self.stock_to_etf_map)
                ou.estimate_industry_beta()
                ou.fit_ou()
                fitted_results[col] = RelativeValueResults(ou)
                m_parameters.append(ou.ou_parameters.m)
            except KeyError:
                excluded.append(col)
            except ValueError:
                excluded.append(col)
            current = i + 1
            print(f"{current}/{total_length} ({current / total_length:.2%}). Excluded: {len(excluded)}",
                  end="\r", flush=True)
        self.fitted_values = fitted_results
        self._mean_m = np.array([m_parameters]).mean()
        self._excluded_stocks = excluded
        self._compute_s_scores()

    def _compute_s_scores(self):
        for entry in self.fitted_values.values():
            centered_m = (entry.ou_process.ou_parameters.m - self._mean_m)
            s_score = -centered_m / entry.ou_process.ou_parameters.sigma_eq
            entry.set_s_score(s_score)
