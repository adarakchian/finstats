from qc.default_imports import *
from qc.stat_arb_stock_etf.constants import EtfStockRvInterface, sector_etf


class EtfStockRvPortfolioConstructor(PortfolioConstructionModel):
    existing_hedges_list: typing.List[Symbol] = []

    def __init__(self, algorithm: EtfStockRvInterface):
        super().__init__()
        self.algorithm = algorithm

    def get_position_data(self, symbol: Symbol, direction: int) -> typing.Dict:
        hedge_ratio = self.algorithm.s_scores.loc[symbol, "coefficient"] * direction * -1
        hedge_etf = self.algorithm.stock_to_etf(symbol)
        return {"stock": symbol, "etf": hedge_etf, "ratio": hedge_ratio, "weight": 0, "direction": direction}

    def get_new_positions(self, insights: typing.List[Insight]) -> pd.DataFrame:
        trading_book = []
        added_symbols = []
        for insight in insights:
            symbol = insight.Symbol
            position_data = self.get_position_data(symbol, insight.Direction)
            trading_book.append(position_data)
            added_symbols.append(symbol)
        trading_book = pd.DataFrame(trading_book)
        if not trading_book.empty:
            trading_book = trading_book.set_index("stock")
        return trading_book

    def parse_existing(self, algorithm: QCAlgorithm) -> (typing.Dict[Symbol, float], typing.List[Symbol], pd.DataFrame):
        scores = self.algorithm.s_scores
        existing_scores = scores.index.tolist()
        portfolio_value = algorithm.Portfolio.TotalPortfolioValue
        hedge_exposures = {}
        positions_to_close = []
        unchanged_positions = []
        portfolio = algorithm.Portfolio
        keys = portfolio.Keys
        investments = [s for s in keys if portfolio[s].Invested]
        for symbol in investments:
            holding = algorithm.Portfolio[symbol]
            if algorithm.Securities[symbol].IsDelisted:
                positions_to_close.append(symbol)
                continue
            holdings_value = holding.HoldingsValue
            holdings_percent = holdings_value / portfolio_value
            direction = 1 if holding.IsLong else -1
            if symbol in self.existing_hedges_list or symbol.Value in sector_etf:
                hedge_exposures[symbol] = holdings_percent
            else:
                if symbol not in existing_scores:
                    positions_to_close.append(symbol)
                    continue
                symbol_score = scores.loc[symbol, "s_score"]
                if holding.IsLong and symbol_score > -self.algorithm.SD_TO_CLOSE_LONG:
                    positions_to_close.append(symbol)
                elif holding.IsShort and symbol_score < self.algorithm.SD_TO_CLOSE_SHORT:
                    positions_to_close.append(symbol)
                else:
                    stock_data = self.get_position_data(symbol, direction)
                    unchanged_positions.append({**stock_data, "weight": np.abs(holdings_percent)})
        unchanged_positions_df = pd.DataFrame(unchanged_positions)
        if not unchanged_positions_df.empty:
            unchanged_positions_df = unchanged_positions_df.set_index("stock")
        return hedge_exposures, positions_to_close, unchanged_positions_df

    def CreateTargets(self, algorithm: QCAlgorithm, insights: typing.List[Insight]) -> typing.List[IPortfolioTarget]:
        # target = PortfolioTarget.Percent(algorithm, "IBM", 0.1)
        abs_scores: pd.Series = self.algorithm.s_scores.loc[:, 's_score'].abs().clip(2, 4)
        max_size = self.algorithm.MAX_NORMAL_POSITION
        rebalance_threshold = max_size * 1

        performance = self.algorithm.stocks_performance

        new_positions = self.get_new_positions(insights)
        new_positions_list: typing.List[Symbol] = new_positions.index.tolist()
        hedge_exposures, positions_to_close, unchanged_positions = self.parse_existing(algorithm)

        # new_portfolio_size = len(new_positions) + len(unchanged_positions)
        # all_positions = pd.concat([new_positions, unchanged_positions])
        # all_positions_list = all_positions.index.tolist()
        if new_positions.empty and len(positions_to_close) == 0:
            return []

        inverse_volatility: pd.Series = 1 / performance.std().loc[new_positions_list]
        total_allowed_weight = max_size * len(inverse_volatility)
        target_weights: pd.Series = inverse_volatility / inverse_volatility.sum()
        target_weights = target_weights.multiply(total_allowed_weight)
        target_weights = target_weights * abs_scores.reindex(target_weights.index).fillna(2).divide(2)
        total_exposure = pd.concat([new_positions, unchanged_positions])
        total_exposure.loc[target_weights.index, "weight"] = target_weights
        total_exposure.loc[:, "weight"] *= total_exposure.loc[:, "direction"]
        total_exposure.loc[:, "etf_weight"] = total_exposure.loc[:, "weight"] * total_exposure.loc[:, "ratio"]
        
        target_stock: typing.Dict[Symbol, float] = total_exposure.loc[target_weights.index, "weight"].to_dict()
        target_hedge: typing.Dict[Symbol, float] = total_exposure.groupby("etf").sum().loc[:, "etf_weight"].to_dict()
        self.existing_hedges_list = list(target_hedge.keys())

        position_dict: typing.Dict[Symbol, float] = {**target_stock, **target_hedge}

        targets = [PortfolioTarget.Percent(algorithm, symbol, weight) for symbol, weight in position_dict.items()]
        for symbol in positions_to_close:
            targets.append(PortfolioTarget.Percent(algorithm, symbol, 0))

        return targets

    def OnSecuritiesChanged(self, algorithm: QCAlgorithm, changes: SecurityChanges) -> None:
        pass