import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import OPTICS
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from hurst import compute_Hc
from arch.unitroot import ADF
import itertools
import typing


class ClusteringPairSelection:
    performance_features: pd.DataFrame = pd.DataFrame()
    explained_variance: pd.Series = pd.Series()
    _clusters: pd.Series = pd.Series()
    pairs_list: typing.List[typing.Tuple]
    cointegrated_pairs_list: typing.List[typing.List]
    cointegration_result: pd.DataFrame = pd.DataFrame()
    filtered_pairs: pd.DataFrame = pd.DataFrame()
    spreads: pd.DataFrame = pd.DataFrame()

    def __init__(self, price: pd.DataFrame):
        price_no_na = price.dropna(axis=1)
        n_dropped = price.shape[1] - price_no_na.shape[1]
        print(f"Dropped {n_dropped} columns out of {price.shape[1]}")
        self.price = price_no_na
        self.log_price = np.log(price_no_na)
        self.performance = self.log_price.diff().iloc[1:]
        self.normal_performance = StandardScaler().fit_transform(self.performance)

    def select_pairs(self):
        print("Converting prices to features...")
        self.returns_to_features(5)
        pd.Series(self.explained_variance).plot(kind='bar', title="Cumulative explained variance")
        plt.show()
        print("Creating clusters....")
        self.create_clusters(3)
        self.clusters.plot(kind='bar', title='Clusters, % of Allocated samples')
        plt.show()
        self.plot_clusters()
        print("Running cointegration check....")
        self.check_cointegration()
        print("Estimating selection criteria...")
        self._calculate_hurst_exponent()
        self._calculate_half_life()
        print("Applying filters...")
        self._apply_post_cointegration_filters()

    def returns_to_features(self, n_components):
        pca = PCA(n_components=n_components)
        transposed_returns = self.normal_performance.T
        pca.fit(transposed_returns)
        reduced_returns = pd.DataFrame(transposed_returns.dot(pca.components_.T), index=self.performance.columns)
        self.explained_variance = pca.explained_variance_ratio_.cumsum()
        self.performance_features = reduced_returns

    def create_clusters(self, min_samples):
        optics = OPTICS(min_samples=min_samples)
        clustering = optics.fit(self.performance_features)
        len(clustering.labels_[clustering.labels_ == -1]) / len(clustering.labels_)
        classified = pd.Series(clustering.labels_, index=self.performance.columns)
        self._clusters = classified
        self._create_cluster_based_pairs()

    @property
    def clusters(self):
        clusters = pd.Series(self._clusters.index.values, index=self._clusters)
        clusters = clusters.groupby(level=0).count()
        clusters /= clusters.sum()
        return clusters

    @staticmethod
    def _npr(n, r=2):
        return np.math.factorial(n) / np.math.factorial(n - r)

    def _create_cluster_based_pairs(self):
        classified = self._clusters[self._clusters != -1]
        all_pairs = []
        for group_id in classified.sort_values().unique():
            group = classified[classified == group_id].index.tolist()
            combinations = list(itertools.permutations(group, 2))
            all_pairs.extend(combinations)
        self.pairs_list = all_pairs

    def check_cointegration(self):
        results = []
        pairs_series = {}
        total_pairs_length = len(self.pairs_list)
        for i, pair in enumerate(self.pairs_list):
            x, y = self.log_price.loc[:, pair].values.T
            pair_name = "|".join(pair)
            pair_id = "|".join(sorted(pair))
            residuals = self._get_residuals(x, y)
            adf_test = ADF(residuals, lags=1)
            p_value = adf_test.pvalue
            test_stat = adf_test.stat
            results.append({"id": pair_id, "p_value": p_value, "stat": test_stat, "pair": pair_name})
            pairs_series[pair_name] = residuals
            current = (i + 1)
            print(f"{current}/{total_pairs_length} ({current / total_pairs_length:.2%})", end="\r", flush=True)
        pairs_series = pd.DataFrame(pairs_series, index=self.price.index)

        results = pd.DataFrame(results).set_index("id")
        results = results.sort_values("p_value", ascending=False).groupby(level=0).first()
        self.cointegration_result = results.set_index("pair")
        valid_pairs = [s.split("|") for s in results.index]
        self.cointegrated_pairs_list = valid_pairs
        self.spreads = pairs_series

    @staticmethod
    def _regress(y, exogenous):
        A = exogenous
        A = np.vstack([np.ones(len(A)), A]).T
        output = np.linalg.inv(A.T @ A) @ A.T @ y
        return output

    @classmethod
    def _get_residuals(cls, x, y):
        intercept, slope = cls._regress(y, x)
        residuals = y - (slope * x + intercept)
        return residuals

    @classmethod
    def _get_half_life(cls, spread):
        change = spread.diff()
        lag = spread.shift().fillna(0)
        intercept, slope = cls._regress(change.iloc[1:], lag.iloc[1:])
        half_life = -np.log(2) / slope
        return half_life

    def _calculate_hurst_exponent(self):
        hurst_values = {}
        for name, row in self.cointegration_result.iterrows():
            pair_ts = self.spreads[name].values
            H, _, _ = compute_Hc(pair_ts)
            hurst_values[name] = H
        hurst_values = pd.Series(hurst_values).rename("hurst")
        self.cointegration_result = self.cointegration_result.join(hurst_values)

    def _calculate_half_life(self):
        half_lives = {}
        for name, row in self.cointegration_result.iterrows():
            pair_ts = self.spreads[name]
            half_lives[name] = self._get_half_life(pair_ts)
        half_lives = pd.Series(half_lives).rename("half_life")
        self.cointegration_result = self.cointegration_result.join(half_lives)

    def _apply_post_cointegration_filters(self):
        self.filtered_pairs = self.cointegration_result[
            (self.cointegration_result.p_value < 0.05) &
            (self.cointegration_result.hurst < 0.5) &
            (self.cointegration_result.half_life > 3) &
            (self.cointegration_result.half_life < 180)
            ]

    def plot_clusters(self, labeled=False):
        tsne = TSNE(learning_rate=1000, perplexity=25).fit_transform(self.performance_features)
        clusters = pd.concat([
            self._clusters.rename("cluster"),
            pd.DataFrame(tsne, columns=["x", "y"], index=self._clusters.index)
        ], axis=1)
        clusters.index = clusters.index.str.replace("(.+)\s.+", r"\1")

        plt.figure(1, facecolor='white')
        plt.clf()
        # plt.axis('off')
        plt.figure(figsize=(20, 10));
        plt.scatter(
            clusters.loc[clusters.cluster != -1, "x"].values,
            clusters.loc[clusters.cluster != -1, "y"].values,
            s=100,
            alpha=0.85,
            c=clusters.loc[clusters.cluster != -1, "cluster"].values,
            cmap=cm.Paired
        )
        plt.scatter(
            clusters.loc[clusters.cluster == -1, "x"].values,
            clusters.loc[clusters.cluster == -1, "y"].values,
            s=100,
            alpha=0.65,
            c="#000"
        )
        if labeled:
            [plt.annotate(name, (row.x, row.y)) for name, row in clusters.iterrows()]

        plt.title('T-SNE with OPTICS clusters')
        plt.show()
