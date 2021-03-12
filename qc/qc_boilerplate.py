# from qc.default_imports import *
# from ou_fit_vectorized import fit_ou_process
#
# sector_etf = ["XLC", "XLY", "XLP", "XLE", "XLF", "XLV", "XLI", "XLB", "XLRE", "XLK", "XLU"]
#
# sector_to_etf_map = {
#     101: "XLB",
#     102: "XLY",
#     103: "XLF",
#     104: "XLRE",
#     205: "XLP",
#     206: "XLV",
#     207: "XLU",
#     308: "XLC",
#     309: "XLE",
#     310: "XLI",
#     311: "XLK",
# }
#
#
# # noinspection PyAbstractClass
# class EtfStockRvInterface(IAlgorithm):
#     closing_prices: typing.Dict[Symbol, RollingWindow]
#     stocks_performance: pd.DataFrame
#     s_scores: pd.DataFrame
#     s_scores_date: typing.Optional[datetime]
#     delisted: typing.List[Symbol]
#
#     SD_TO_OPEN: float
#     SD_TO_CLOSE_LONG: float
#     SD_TO_CLOSE_SHORT: float
#     MAX_NORMAL_POSITION: float
#     OPTIMAL_NET_LONG: float
#
#     def stock_to_etf(self, stock_symbol: Symbol) -> Symbol:
#         raise NotImplemented("Override this method")
#
#
# class EtfStockRvAlphaModel(AlphaModel):
#     def __init__(self, algorithm: EtfStockRvInterface):
#         super().__init__()
#         self.Name = f"S-Score-based RV Alpha"
#         self.algorithm = algorithm
#
#     def Update(self, algorithm: QCAlgorithm, data: Slice) -> typing.List[Insight]:
#         s_scores = self.algorithm.s_scores
#         portfolio = self.algorithm.Portfolio
#         keys = portfolio.Keys
#         investments_list = [s for s in keys if portfolio[s].Invested]
#
#         s_scores = s_scores.loc[~s_scores.index.isin([*investments_list])]
#         filtered = s_scores.loc[
#             (s_scores.k > 252 / 30) &
#             # (s_scores.tau < 0.2 * (60 / 252)) &
#             (s_scores.s_score.abs() > 2)
#             ]
#         long_symbols = filtered[filtered.s_score < 0].index.tolist()
#         short_symbols = filtered[filtered.s_score > 0].index.tolist()
#         longs = [Insight.Price(symbol, timedelta(days=3), InsightDirection.Up) for symbol in long_symbols]
#         shorts = [Insight.Price(symbol, timedelta(days=3), InsightDirection.Down) for symbol in short_symbols]
#         insights: typing.List[Insight] = [*longs, *shorts]
#
#         filtered_insights = [s for s in insights if not algorithm.Securities[s.Symbol].IsDelisted]
#         return filtered_insights
#
#     def OnSecuritiesChanged(self, algorithm: QCAlgorithm, changes: SecurityChanges) -> None:
#         pass
#
#
# class EtfStockRvPortfolioConstructor(PortfolioConstructionModel):
#     existing_hedges_list: typing.List[Symbol] = []
#
#     def __init__(self, algorithm: EtfStockRvInterface):
#         super().__init__()
#         self.algorithm = algorithm
#
#     def get_position_data(self, symbol: Symbol, direction: int) -> typing.Dict:
#         hedge_ratio = self.algorithm.s_scores.loc[symbol, "coefficient"] * direction * -1
#         hedge_etf = self.algorithm.stock_to_etf(symbol)
#         return {"stock": symbol, "etf": hedge_etf, "ratio": hedge_ratio, "weight": 0}
#
#     def get_new_positions(self, insights: typing.List[Insight]) -> pd.DataFrame:
#         trading_book = []
#         added_symbols = []
#         for insight in insights:
#             symbol = insight.Symbol
#             position_data = self.get_position_data(symbol, insight.Direction)
#             trading_book.append(position_data)
#             added_symbols.append(symbol)
#         trading_book = pd.DataFrame(trading_book)
#         if not trading_book.empty:
#             trading_book = trading_book.set_index("stock")
#         return trading_book
#
#     def parse_existing(self, algorithm: QCAlgorithm) -> (typing.Dict[Symbol, float], typing.List[Symbol], pd.DataFrame):
#         scores = self.algorithm.s_scores
#         existing_scores = scores.index.tolist()
#         portfolio_value = self.algorithm.Portfolio.TotalPortfolioValue
#         hedge_exposures = {}
#         positions_to_close = []
#         unchanged_positions = []
#         portfolio = self.algorithm.Portfolio
#         keys = portfolio.Keys
#         investments = [s for s in keys if portfolio[s].Invested]
#         for symbol in investments:
#             holding = self.algorithm.Portfolio[symbol]
#             if algorithm.Securities[symbol].IsDelisted:
#                 positions_to_close.append(symbol)
#                 continue
#             holdings_value = holding.HoldingsValue
#             holdings_percent = holdings_value / portfolio_value
#             direction = 1 if holding.IsLong else -1
#             if symbol in self.existing_hedges_list or symbol.Value in sector_etf:
#                 hedge_exposures[symbol] = holdings_percent
#             else:
#                 if symbol not in existing_scores:
#                     positions_to_close.append(symbol)
#                     continue
#                 symbol_score = scores.loc[symbol, "s_score"]
#                 if holding.IsLong and symbol_score > -self.algorithm.SD_TO_CLOSE_LONG:
#                     positions_to_close.append(symbol)
#                 elif holding.IsShort and symbol_score < self.algorithm.SD_TO_CLOSE_SHORT:
#                     positions_to_close.append(symbol)
#                 else:
#                     stock_data = self.get_position_data(symbol, direction)
#                     unchanged_positions.append({**stock_data, "weight": holdings_percent})
#         unchanged_positions_df = pd.DataFrame(unchanged_positions)
#         if not unchanged_positions_df.empty:
#             unchanged_positions_df = unchanged_positions_df.set_index("stock")
#         return hedge_exposures, positions_to_close, unchanged_positions_df
#
#     def CreateTargets(self, algorithm: QCAlgorithm, insights: typing.List[Insight]) -> typing.List[IPortfolioTarget]:
#         # target = PortfolioTarget.Percent(algorithm, "IBM", 0.1)
#         abs_scores: pd.Series = self.algorithm.s_scores.loc[:, 's_score'].abs().clip(2, 4)
#         max_size = self.algorithm.MAX_NORMAL_POSITION
#
#         performance = self.algorithm.stocks_performance
#
#         new_positions = self.get_new_positions(insights)
#         hedge_exposures, positions_to_close, unchanged_positions = self.parse_existing(algorithm)
#
#         # new_portfolio_size = len(new_positions) + len(unchanged_positions)
#         all_positions = pd.concat([new_positions, unchanged_positions])
#         all_positions_list = all_positions.index.tolist()
#         if all_positions.empty:
#             return []
#         inverse_volatility: pd.Series = 1 / performance.std().loc[all_positions_list]
#         total_allowed_weight = max_size * len(inverse_volatility)
#         target_weights: pd.Series = inverse_volatility / inverse_volatility.sum()
#         target_weights = target_weights.multiply(total_allowed_weight)
#         target_weights = target_weights * abs_scores.reindex(target_weights.index).fillna(2).divide(2)
#         # if target_weights.sum() < self.algorithm.OPTIMAL_NET_LONG:
#         #     target_weights = (target_weights / target_weights.sum()) * self.algorithm.OPTIMAL_NET_LONG
#         weight_difference = target_weights.subtract(all_positions.loc[:, "weight"])
#
#         positions_to_change = weight_difference[weight_difference.abs() > max_size].index.tolist()
#         # positions_to_keep = weight_difference[weight_difference.abs() <= max_size].index.tolist()
#         all_positions.loc[positions_to_change, "weight"] = target_weights.loc[positions_to_change]
#         all_positions.loc[:, "etf_weight"] = all_positions.loc[:, "weight"] * all_positions.loc[:, "ratio"]
#         target_stock: typing.Dict[Symbol, float] = all_positions.loc[:, 'weight'].to_dict()
#         target_hedge: typing.Dict[Symbol, float] = all_positions.groupby("etf").sum().loc[:, "etf_weight"].to_dict()
#         self.existing_hedges_list = list(target_hedge.keys())
#
#         position_dict: typing.Dict[Symbol, float] = {**target_stock, **target_hedge}
#
#         targets = [PortfolioTarget.Percent(algorithm, symbol, weight) for symbol, weight in position_dict.items()]
#         for symbol in [*positions_to_close]:
#             targets.append(PortfolioTarget.Percent(algorithm, symbol, 0))
#
#         return targets
#
#     def OnSecuritiesChanged(self, algorithm: QCAlgorithm, changes: SecurityChanges) -> None:
#         pass
#
#
# class EtfStockRvAlgo(QCAlgorithm):
#     ROLLING = 60
#     SD_TO_OPEN = 2
#     SD_TO_CLOSE_LONG = 0.75
#     SD_TO_CLOSE_SHORT = 0.5
#     MAX_NORMAL_POSITION = 3 / 100
#     OPTIMAL_NET_LONG = 0.8
#
#     _stock_to_etf_ticker_map: typing.Dict[Symbol, str] = {}
#     _etf_ticker_to_etf_symbol: typing.Dict[str, Symbol] = {}
#
#     non_stocks: typing.List[str] = ["SPY", *sector_to_etf_map.values()]
#     stocks: typing.List[Symbol] = []
#
#     closing_prices: typing.Dict[Symbol, RollingWindow] = {}
#     stocks_performance: pd.DataFrame = pd.DataFrame()
#
#     s_scores: pd.DataFrame = pd.DataFrame()
#     s_scores_date: typing.Optional[datetime] = None
#
#     def Initialize(self):
#         self.SetStartDate(2019, 7, 1)  # Set Start Date
#         self.SetEndDate(2020, 11, 30)  # Set End Date
#         self.SetCash(1000000)  # Set Strategy Cash
#
#         self.AddEquity("SPY", Resolution.Daily)
#         [self.AddEquity(s, Resolution.Daily) for s in sector_etf]
#         self.AddUniverse(self.Universe.Index.QC500)
#
#         self.UniverseSettings.Resolution = Resolution.Daily
#
#         # noinspection PyTypeChecker
#         self.AddAlpha(EtfStockRvAlphaModel(self))
#         # noinspection PyTypeChecker
#         self.SetPortfolioConstruction(EtfStockRvPortfolioConstructor(self))
#
#         self.add_exposures_chart()
#
#         self.Schedule.On(self.DateRules.EveryDay("SPY"), self.TimeRules.AfterMarketOpen("SPY"), self.report)
#
#     def add_exposures_chart(self):
#         exposures = Chart("Exposures")
#         exposures.AddSeries(Series("Long", SeriesType.Line, 0, "%"))
#         exposures.AddSeries(Series("Short", SeriesType.Line, 0, "%"))
#         exposures.AddSeries(Series("Net", SeriesType.Line, 0, "%"))
#         self.AddChart(exposures)
#
#     def plot_exposures(self):
#         portfolio_value = self.Portfolio.TotalPortfolioValue
#         keys = self.Portfolio.Keys
#         investments_list = [s for s in keys if self.Portfolio[s].Invested]
#         longs = []
#         shorts = []
#         for s in investments_list:
#             holding = self.Portfolio[s]
#             exposure = holding.HoldingsValue / portfolio_value
#             if exposure > 0:
#                 longs.append(exposure)
#             else:
#                 shorts.append(exposure)
#         long = np.array(longs).sum() * 100
#         short = np.array(shorts).sum() * 100
#         self.Debug(f"Longs: {long}")
#         self.Debug(f"Shorts: {short}")
#         net = long + short
#
#         self.Plot("Exposures", "Long", long)
#         self.Plot("Exposures", "Short", short)
#         self.Plot("Exposures", "Net", net)
#
#     def OnData(self, data: Slice) -> None:
#         # self.Debug(f"Status update on {self.Time}")
#         self.prepare_data()
#         self.plot_exposures()
#
#     def report(self):
#         pass
#
#     def OnSecuritiesChanged(self, changes: SecurityChanges):
#         for security in changes.AddedSecurities:
#             symbol = security.Symbol
#             self.update_etf_map(security)
#             self.closing_prices[symbol] = RollingWindow[float](self.ROLLING)
#             if symbol.Value in self.non_stocks:
#                 self._etf_ticker_to_etf_symbol[symbol.Value] = symbol
#
#         added_symbols = [s.Symbol for s in changes.AddedSecurities]
#         self.warmup_closing_prices(added_symbols)
#         for security in changes.RemovedSecurities:
#             symbol = security.Symbol
#             if symbol in self.closing_prices.keys():
#                 self.closing_prices[symbol].Reset()
#                 self.closing_prices.pop(symbol)
#
#     def warmup_closing_prices(self, symbols: typing.List[Symbol]):
#         history = self.History(symbols, self.ROLLING, Resolution.Daily)
#         for symbol in symbols:
#             for time, row in history.loc[symbol].iterrows():
#                 self.closing_prices[symbol].Add(row['close'])
#
#     def update_etf_map(self, security: Security):
#         symbol = security.Symbol
#         if symbol.Value in self.non_stocks:
#             return
#         fundamentals = security.Fundamentals
#         if not fundamentals:
#             self.Debug(f"No fundamentals for {symbol.Value}")
#             return
#         sector_code = fundamentals.AssetClassification.MorningstarSectorCode
#         etf_ticker = sector_to_etf_map[sector_code]
#         self._stock_to_etf_ticker_map[symbol] = etf_ticker
#         if symbol not in self.stocks:
#             self.stocks.append(symbol)
#
#     def prepare_data(self):
#         ready_prices: typing.Dict[Symbol, RollingWindow] = {}
#         for key, value in self.closing_prices.items():
#             if value.IsReady and value.Count == self.ROLLING:
#                 ready_prices[key] = value
#             else:
#                 pass
#         prices = pd.DataFrame(ready_prices)
#         performance: pd.DataFrame = np.log(prices).diff().iloc[1:]
#         performance = performance.dropna(axis=1)
#
#         available_stocks = [s for s in self.stocks if s in performance.columns]
#
#         stocks_performance: pd.DataFrame = performance.loc[:, available_stocks].dropna(axis=1)
#         stocks_tickers = [s for s in available_stocks if s in stocks_performance.columns]
#         benchmark_symbols = [self.stock_to_etf(s) for s in stocks_tickers]
#         benchmarks_performance: pd.DataFrame = performance.loc[:, benchmark_symbols]
#         self.s_scores = fit_ou_process(stocks_performance, benchmarks_performance)
#         self.s_scores_date = self.Time
#         self.stocks_performance = stocks_performance
#
#     def stock_to_etf(self, stock_symbol: Symbol):
#         etf_ticker = self._stock_to_etf_ticker_map[stock_symbol]
#         return self._etf_ticker_to_etf_symbol[etf_ticker]
