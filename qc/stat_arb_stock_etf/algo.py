from qc.default_imports import *
from stats.ou_fit_vectorized import fit_ou_process
from qc.stat_arb_stock_etf.constants import sector_etf, sector_to_etf_map
from qc.stat_arb_stock_etf.alpha import EtfStockRvAlphaModel
from qc.stat_arb_stock_etf.portfolio import EtfStockRvPortfolioConstructor


class EtfStockRvAlgo(QCAlgorithm):
    ROLLING = 60
    SD_TO_OPEN = 2
    SD_TO_CLOSE_LONG = 0.75
    SD_TO_CLOSE_SHORT = 0.5
    MAX_NORMAL_POSITION = 3 / 100
    OPTIMAL_NET_LONG = 0.8

    _stock_to_etf_ticker_map: typing.Dict[Symbol, str] = {}
    _etf_ticker_to_etf_symbol: typing.Dict[str, Symbol] = {}

    non_stocks: typing.List[str] = ["SPY", *sector_to_etf_map.values()]
    stocks: typing.List[Symbol] = []

    closing_prices: typing.Dict[Symbol, RollingWindow] = {}
    stocks_performance: pd.DataFrame = pd.DataFrame()

    s_scores: pd.DataFrame = pd.DataFrame()
    s_scores_date: typing.Optional[datetime] = None

    def Initialize(self):
        self.SetStartDate(2019, 1, 1)  # Set Start Date
        self.SetEndDate(2020, 11, 30)  # Set End Date
        self.SetCash(1000000)  # Set Strategy Cash

        self.AddEquity("SPY", Resolution.Daily)
        [self.AddEquity(s, Resolution.Daily) for s in sector_etf]
        self.AddUniverse(self.Universe.Index.QC500)

        self.UniverseSettings.Resolution = Resolution.Daily
        self.UniverseSettings.Leverage = 3

        # noinspection PyTypeChecker
        self.AddAlpha(EtfStockRvAlphaModel(self))
        # noinspection PyTypeChecker
        self.SetPortfolioConstruction(EtfStockRvPortfolioConstructor(self))

        self.add_exposures_chart()

        self.Schedule.On(self.DateRules.EveryDay("SPY"), self.TimeRules.AfterMarketOpen("SPY"), self.report)

    def add_exposures_chart(self):
        exposures = Chart("Exposures")
        exposures.AddSeries(Series("Long", SeriesType.Line, 0, "%"))
        exposures.AddSeries(Series("Short", SeriesType.Line, 0, "%"))
        exposures.AddSeries(Series("Net", SeriesType.Line, 0, "%"))
        self.AddChart(exposures)

    def plot_exposures(self):
        portfolio_value = self.Portfolio.TotalPortfolioValue
        keys = self.Portfolio.Keys
        investments_list = [s for s in keys if self.Portfolio[s].Invested]
        longs = []
        shorts = []
        for s in investments_list:
            holding = self.Portfolio[s]
            exposure = holding.HoldingsValue / portfolio_value
            if exposure > 0:
                longs.append(exposure)
            else:
                shorts.append(exposure)
        long = np.array(longs).sum() * 100
        short = np.array(shorts).sum() * 100
        net = long + short

        self.Plot("Exposures", "Long", long)
        self.Plot("Exposures", "Short", short)
        self.Plot("Exposures", "Net", net)

    def OnData(self, data: Slice) -> None:
        # self.Debug(f"Status update on {self.Time}")
        self.prepare_data()
        self.plot_exposures()

    def report(self):
        pass

    def OnSecuritiesChanged(self, changes: SecurityChanges):
        for security in changes.AddedSecurities:
            symbol = security.Symbol
            self.update_etf_map(security)
            self.closing_prices[symbol] = RollingWindow[float](self.ROLLING)
            if symbol.Value in self.non_stocks:
                self._etf_ticker_to_etf_symbol[symbol.Value] = symbol

        added_symbols = [s.Symbol for s in changes.AddedSecurities]
        self.warmup_closing_prices(added_symbols)
        for security in changes.RemovedSecurities:
            symbol = security.Symbol
            if symbol in self.closing_prices.keys():
                self.closing_prices[symbol].Reset()
                self.closing_prices.pop(symbol)

    def warmup_closing_prices(self, symbols: typing.List[Symbol]):
        history = self.History(symbols, self.ROLLING, Resolution.Daily)
        for symbol in symbols:
            for time, row in history.loc[symbol].iterrows():
                self.closing_prices[symbol].Add(row['close'])

    def update_etf_map(self, security: Security):
        symbol = security.Symbol
        if symbol.Value in self.non_stocks:
            return
        fundamentals = security.Fundamentals
        if not fundamentals:
            self.Debug(f"No fundamentals for {symbol.Value}")
            return
        sector_code = fundamentals.AssetClassification.MorningstarSectorCode
        etf_ticker = sector_to_etf_map[sector_code]
        self._stock_to_etf_ticker_map[symbol] = etf_ticker
        if symbol not in self.stocks:
            self.stocks.append(symbol)

    def prepare_data(self):
        ready_prices: typing.Dict[Symbol, RollingWindow] = {}
        for key, value in self.closing_prices.items():
            if value.IsReady and value.Count == self.ROLLING:
                ready_prices[key] = value
            else:
                pass
        prices = pd.DataFrame(ready_prices)
        performance: pd.DataFrame = np.log(prices).diff().iloc[1:]
        performance = performance.dropna(axis=1)

        available_stocks = [s for s in self.stocks if s in performance.columns]

        stocks_performance: pd.DataFrame = performance.loc[:, available_stocks].dropna(axis=1)
        stocks_tickers = [s for s in available_stocks if s in stocks_performance.columns]
        benchmark_symbols = [self.stock_to_etf(s) for s in stocks_tickers]
        benchmarks_performance: pd.DataFrame = performance.loc[:, benchmark_symbols]
        self.s_scores = fit_ou_process(stocks_performance, benchmarks_performance)
        self.s_scores_date = self.Time
        self.stocks_performance = stocks_performance

    def stock_to_etf(self, stock_symbol: Symbol):
        etf_ticker = self._stock_to_etf_ticker_map[stock_symbol]
        return self._etf_ticker_to_etf_symbol[etf_ticker]
