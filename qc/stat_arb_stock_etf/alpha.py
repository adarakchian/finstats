from qc.default_imports import *
from qc.stat_arb_stock_etf.constants import EtfStockRvInterface


class EtfStockRvAlphaModel(AlphaModel):
    def __init__(self, algorithm: EtfStockRvInterface):
        super().__init__()
        self.Name = f"S-Score-based RV Alpha"
        self.algorithm = algorithm

    def Update(self, algorithm: QCAlgorithm, data: Slice) -> typing.List[Insight]:
        s_scores = self.algorithm.s_scores
        portfolio = self.algorithm.Portfolio
        keys = portfolio.Keys
        investments_list = [s for s in keys if portfolio[s].Invested]

        s_scores = s_scores.loc[~s_scores.index.isin([*investments_list])]
        filtered = s_scores.loc[
            (s_scores.k > 252 / 30) &
            # (s_scores.tau < 0.2 * (60 / 252)) &
            (s_scores.s_score.abs() > 2)
            ]
        long_symbols = filtered[filtered.s_score < 0].index.tolist()
        short_symbols = filtered[filtered.s_score > 0].index.tolist()
        longs = [Insight.Price(symbol, timedelta(days=3), InsightDirection.Up) for symbol in long_symbols]
        shorts = [Insight.Price(symbol, timedelta(days=3), InsightDirection.Down) for symbol in short_symbols]
        insights: typing.List[Insight] = [*longs, *shorts]

        filtered_insights = [s for s in insights if not algorithm.Securities[s.Symbol].IsDelisted]
        return filtered_insights

    def OnSecuritiesChanged(self, algorithm: QCAlgorithm, changes: SecurityChanges) -> None:
        pass
