from default_imports import *
from stats.ou_fit_vectorized import fit_ou_process
import pandas as pd
import numpy as np
import typing


# noinspection DuplicatedCode
class StatisticalArbitrage(QCAlgorithm):
    LOOKBACK: int = 60
    MAX_POSITIONS: int = 120
    SD_TO_OPEN = 2
    SD_TO_CLOSE_LONG = 0.75
    SD_TO_CLOSE_SHORT = 0.5

    _implied_position_weight: float

    symbol_to_etf: typing.Dict[Symbol, str] = {}
    closing_prices: typing.Dict[Symbol, RollingWindow] = {}
    sector_to_etf_map = {
        101: "XLB",
        102: "XLY",
        103: "XLF",
        104: "XLRE",
        205: "XLP",
        206: "XLV",
        207: "XLU",
        308: "XLC",
        309: "XLE",
        310: "XLI",
        311: "XLK",
        999: "SPY"
    }

    def Initialize(self):
        self.SetStartDate(2019, 1, 1)
        self.SetEndDate(2019, 12, 31)
        self.SetCash(10000000)

        self.AddEquity("SPY", Resolution.Daily)
        [self.AddEquity(s, Resolution.Daily) for s in self.sector_to_etf_map.values()]
        self.AddUniverse(self.Universe.Index.QC500)

        self.UniverseSettings.Resolution = Resolution.Daily
        self.UniverseSettings.MinimumTimeInUniverse = timedelta(days=90)

        self._implied_position_weight = 0.8 / self.MAX_POSITIONS

    def OnData(self, data: Slice):
        performance = self.fetch_performance()
        stocks, benchmarks = self.split_stock_benchmark(performance)
        scores = fit_ou_process(stocks, benchmarks)
        filtered = self.create_insights(scores)
        self.handle_trades(scores, filtered)
        pass

    def OnSecuritiesChanged(self, changes: SecurityChanges):
        for security in changes.AddedSecurities:
            symbol = security.Symbol
            self.closing_prices[symbol] = RollingWindow[float](self.LOOKBACK)

            fundamentals = security.Fundamentals
            if not fundamentals:
                continue
            sector_code = fundamentals.AssetClassification.MorningstarSectorCode
            etf = self.sector_to_etf_map[sector_code]
            self.symbol_to_etf[symbol] = etf
        added_symbols = [s.Symbol for s in changes.AddedSecurities]
        self.warmup_closing_prices(added_symbols)
        for security in changes.RemovedSecurities:
            symbol = security.Symbol
            if symbol in self.closing_prices.keys():
                self.closing_prices[symbol].Reset()
                self.closing_prices.pop(symbol)

    def warmup_closing_prices(self, symbols: typing.List[Symbol]):
        history = self.History(symbols, self.LOOKBACK, Resolution.Daily)
        for symbol in symbols:
            for time, row in history.loc[symbol].iterrows():
                self.closing_prices[symbol].Add(row['close'])

    def fetch_performance(self) -> pd.DataFrame:
        # keys = self.Securities.Keys
        # history: pd.DataFrame = self.History(keys, self.LOOKBACK, Resolution.Daily).close
        # prices: pd.DataFrame = history.unstack(level=0).dropna(axis=1)
        ready_prices: typing.Dict[Symbol, RollingWindow] = {}
        for key, value in self.closing_prices.items():
            if value.IsReady and value.Count == self.LOOKBACK:
                ready_prices[key] = value
            else:
                pass
        prices = {str(k): v for k, v in ready_prices.items()}
        prices = pd.DataFrame(prices)
        performance: pd.DataFrame = np.log(prices).diff().iloc[1:]
        return performance

    def split_stock_benchmark(self, performance):
        keys = self.Securities.Keys
        etf_names = [*self.sector_to_etf_map.values()]
        etf_filter = performance.columns.isin(etf_names)
        etfs: pd.DataFrame = performance.loc[:, etf_filter]
        stocks: pd.DataFrame = performance.loc[:, ~etf_filter]
        symbols = {str(i): i for i in keys if str(i) in stocks.columns}
        benchmark_list = [self.symbol_to_etf[symbols[s]] for s in stocks.columns]
        benchmarks = etfs.loc[:, benchmark_list]
        return stocks, benchmarks

    def create_insights(self, score: pd.DataFrame):
        securities = self.Portfolio.Keys
        investments = [s for s in securities if self.Portfolio[s].Invested]
        filtered = score.loc[
            (score.k > 252 / 30) &
            (score.s_score.abs() > 2) &
            (~score.index.isin(investments))
            ]
        long_symbols = filtered[filtered.s_score < 0].index.tolist()
        short_symbols = filtered[filtered.s_score > 0].index.tolist()
        long = [Insight.Price(symbol, timedelta(days=3), InsightDirection.Up) for symbol in long_symbols]
        short = [Insight.Price(symbol, timedelta(days=3), InsightDirection.Down) for symbol in short_symbols]
        insights = [*long, *short]
        self.EmitInsights(insights)
        return filtered

    def handle_trades(self, full, ou_parameters: pd.DataFrame):
        score: pd.Series = ou_parameters.loc[:, 's_score']
        new_positions_length = len(score)
        # stocks, hedges = self.investments
        closures = self.find_converged(full)
        stock_weights = self.stock_weights
        nominal_weights: pd.Series = (score / score.abs().sum()).sort_values(ascending=False)
        max_new_positions = self.MAX_POSITIONS - len(stock_weights) + len(closures)
        cutoff = min(new_positions_length, max_new_positions)
        real_weights: typing.Dict = (nominal_weights.iloc[:cutoff] * self._implied_position_weight * cutoff).to_dict()
        [self.SetHoldings(s, 0) for s in closures]
        [self.SetHoldings(s, w) for s, w in real_weights.items()]
        return

    @property
    def investments(self) -> (typing.List[Symbol], typing.List[Symbol]):
        etfs = self.sector_to_etf_map.values()
        securities = self.Portfolio.Keys
        investments = [s for s in securities if self.Portfolio[s].Invested]
        stocks = [s for s in investments if s.Value in etfs]
        hedges = [s for s in investments if s.Value not in etfs]
        return stocks, hedges

    @property
    def stock_weights(self):
        stocks, _ = self.investments
        total_value = self.Portfolio.TotalPortfolioValue
        position_info = []
        for s in stocks:
            position_info.append({
                "stock": s,
                "dollar_value": self.Portfolio[s].HoldingsValue,
                "absolute_value": self.Portfolio[s].AbsoluteHoldingsValue
            })
        positions = pd.DataFrame(position_info)
        if positions.empty:
            return pd.DataFrame()
        positions.loc[:, 'weight'] = positions.loc[:, 'dollar_value'] / total_value
        positions.loc[:, 'absolute_weight'] = positions.loc[:, 'absolute_value'] / total_value
        positions = positions.set_index("stock")
        return positions

    def find_converged(self, score: pd.Series):
        stocks, _ = self.investments
        positions_to_close = []
        for symbol in stocks:
            holding = self.Portfolio[symbol]
            if symbol not in score.index:
                positions_to_close.append(symbol)
                continue
            symbol_score = score.loc[symbol]
            long_condition = holding.IsLong and symbol_score > -self.SD_TO_CLOSE_LONG
            short_condition = holding.IsShort and symbol_score < self.SD_TO_CLOSE_SHORT
            if long_condition or short_condition:
                positions_to_close.append(symbol)
        return positions_to_close
