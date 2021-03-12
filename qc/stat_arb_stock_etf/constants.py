from qc.default_imports import *


sector_etf = ["XLC", "XLY", "XLP", "XLE", "XLF", "XLV", "XLI", "XLB", "XLRE", "XLK", "XLU"]


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
}


# noinspection PyAbstractClass
class EtfStockRvInterface(IAlgorithm):
    closing_prices: typing.Dict[Symbol, RollingWindow]
    stocks_performance: pd.DataFrame
    s_scores: pd.DataFrame
    s_scores_date: typing.Optional[datetime]
    delisted: typing.List[Symbol]

    SD_TO_OPEN: float
    SD_TO_CLOSE_LONG: float
    SD_TO_CLOSE_SHORT: float
    MAX_NORMAL_POSITION: float
    OPTIMAL_NET_LONG: float

    def stock_to_etf(self, stock_symbol: Symbol) -> Symbol:
        raise NotImplemented("Override this method")
