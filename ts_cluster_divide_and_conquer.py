# ts_cluster_divide_and_conquer.py
#Required libraries
import numpy as np
import math
from typing import List, Tuple
import matplotlib.pyplot as plt
import time
import random

# Utilities: DTW
def dtw_distance(a: np.ndarray, b: np.ndarray, window: int = None, early_abort: float = None) -> float:
    
    #Classic DTW (no pruning by default). Optionally use Sakoe-Chiba window.
    #early_abort: if partial path cost exceeds this, return +inf for early abandoning.
    n, m = len(a), len(b)
    if window is None:
        window = max(n, m)
    window = max(window, abs(n - m))
    INF = float('inf')
    D = np.full((n+1, m+1), INF)
    D[0,0] = 0.0
    for i in range(1, n+1):
        start = max(1, i - window)
        end = min(m, i + window)
        ai = a[i-1]
        row_min = INF
        for j in range(start, end+1):
            cost = (ai - b[j-1])**2
            val = cost + min(D[i-1,j], D[i,j-1], D[i-1,j-1])
            D[i,j] = val
            if val < row_min:
                row_min = val
        if early_abort is not None and row_min > early_abort:
            return float('inf')
    return math.sqrt(D[n,m])

# Kadane (max subarray) on activity
def kadane(arr: np.ndarray) -> Tuple[int,int,float]:
    #Returns (start_idx, end_idx_inclusive, max_sum)
    max_ending = arr[0]
    max_so_far = arr[0]
    s = 0
    start = 0
    end = 0
    for i in range(1, len(arr)):
        if arr[i] > max_ending + arr[i]:
            max_ending = arr[i]
            s = i
        else:
            max_ending = max_ending + arr[i]
        if max_ending > max_so_far:
            max_so_far = max_ending
            start = s
            end = i
    return start, end, max_so_far

# Closest pair brute-force (can be optimized)
def closest_pair_indices(series_list: List[np.ndarray], window: int = None) -> Tuple[Tuple[int,int], float]:
    best = (None, None)
    best_dist = float('inf')
    n = len(series_list)
    for i in range(n):
        for j in range(i+1, n):
            d = dtw_distance(series_list[i], series_list[j], window=window, early_abort=best_dist)
            if d < best_dist:
                best_dist = d
                best = (i, j)
    return best, best_dist

# Divisive clustering
class DivisiveClustering:
    def __init__(self, series_list: List[np.ndarray], min_cluster_size: int = 5, max_depth: int = 10,
                 dtw_window: int = None, variance_threshold: float = 1e-6):
        self.series_list = series_list
        self.min_cluster_size = min_cluster_size
        self.max_depth = max_depth
        self.window = dtw_window
        self.variance_threshold = variance_threshold
        self.clusters = []  # list of lists of indices

    def run(self):
        indices = list(range(len(self.series_list)))
        self._recurse(indices, depth=0)
        return self.clusters

    def _cluster_variance(self, indices: List[int]) -> float:
        #Simple measure: variance across flattened z-scored segments (cheap proxy).
        if len(indices) <= 1:
            return 0.0
        arr = np.stack([self.series_list[i] for i in indices])
        s = np.std(arr, axis=0)
        return float(np.mean(s))  # mean of per-timestep stds

    def _farthest_pair_seeds(self, indices: List[int]) -> Tuple[int,int]:
        #Farthest-first seeds: pick idx a -> b farthest from a -> c farthest from b -> return (b,c).
        a = random.choice(indices)
        # compute distances from a
        best_b, best_d = a, -1.0
        for i in indices:
            d = dtw_distance(self.series_list[a], self.series_list[i], window=self.window)
            if d > best_d:
                best_d = d
                best_b = i
        best_c, best_d2 = best_b, -1.0
        for i in indices:
            d = dtw_distance(self.series_list[best_b], self.series_list[i], window=self.window)
            if d > best_d2:
                best_d2 = d
                best_c = i
        return best_b, best_c

    def _recurse(self, indices: List[int], depth: int):
        if len(indices) <= self.min_cluster_size or depth >= self.max_depth:
            self.clusters.append(indices)
            return
        # stopping based on variance
        var = self._cluster_variance(indices)
        if var < self.variance_threshold:
            self.clusters.append(indices)
            return

        # pick seeds
        seed1, seed2 = self._farthest_pair_seeds(indices)
        group1, group2 = [], []
        for idx in indices:
            d1 = dtw_distance(self.series_list[idx], self.series_list[seed1], window=self.window, early_abort=None)
            d2 = dtw_distance(self.series_list[idx], self.series_list[seed2], window=self.window, early_abort=None)
            if d1 <= d2:
                group1.append(idx)
            else:
                group2.append(idx)

        # if a trivial split occurred, stop
        if len(group1) == 0 or len(group2) == 0:
            self.clusters.append(indices)
            return

        # recurse
        self._recurse(group1, depth+1)
        self._recurse(group2, depth+1)


# Visualization helpers
def plot_cluster_representatives(series_list: List[np.ndarray], clusters: List[List[int]], show_n=3):
    for ci, cluster in enumerate(clusters):
        plt.figure(figsize=(8,3))
        plt.title(f"Cluster {ci} — size {len(cluster)}")
        for i, idx in enumerate(cluster[:show_n]):
            plt.plot(series_list[idx], alpha=0.9, label=f'idx {idx}')
        plt.legend()
        plt.tight_layout()
        plt.show()

# Toy demo / verification
def generate_toy_signals(n=50, length=200, seed=0):
    np.random.seed(seed)
    base1 = np.sin(np.linspace(0, 6*np.pi, length))
    base2 = np.sign(np.sin(np.linspace(0, 3*np.pi, length))) * np.exp(-np.linspace(0,2,length))
    data = []
    for i in range(n):
        if i < n//2:
            data.append(base1 + 0.1*np.random.randn(length))
        else:
            data.append(base2 + 0.1*np.random.randn(length))
    return [np.array(x) for x in data]

def demo_toy():
    data = generate_toy_signals(40, 200)
    dc = DivisiveClustering(data, min_cluster_size=4, max_depth=6, dtw_window=20)
    t0 = time.time()
    clusters = dc.run()
    t1 = time.time()
    print(f"Found {len(clusters)} clusters in {t1-t0:.2f}s")
    plot_cluster_representatives(data, clusters, show_n=3)
    # closest pair & Kadane per cluster
    for ci, cluster in enumerate(clusters):
        subseries = [data[i] for i in cluster]
        (i,j), dist = closest_pair_indices(subseries, window=20)
        print(f"Cluster {ci} size {len(cluster)}: closest pair indices (relative) {i},{j} dist {dist:.3f}")
        # Kadane on abs diff for each series:
        for rel_idx, idx in enumerate(cluster[:2]):  # show first two
            s = data[idx]
            activity = np.abs(np.diff(s))
            st, ed, val = kadane(activity)
            print(f"  series {idx}: kadane interval [{st},{ed}] sum={val:.3f}")
            # optional plot overlay
            plt.figure(figsize=(6,2))
            plt.plot(s, label='signal')
            plt.axvspan(st, ed+1, alpha=0.2, label='Kadane activity window')
            plt.legend()
            plt.title(f'Cluster {ci} series {idx}')
            plt.tight_layout()
            plt.show()

#Executes code
if __name__ == "__main__":
    demo_toy()




