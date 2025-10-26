# PulseDB.py
import h5py
import numpy as np
import random
from ts_cluster_divide_and_conquer import DivisiveClustering, closest_pair_indices, kadane, plot_cluster_representatives

# 1. Load PulseDB dataset
file_path = r'J:\PyCharmProjects\PythonProject\VitalDB_CalBased_Test_Subset.mat'

all_abp_segments = []
all_demographics = []

with h5py.File(file_path, 'r') as f:
    subset = f['Subset']
    signals = subset['Signals']
    num_segments_raw = signals.shape[0]

    # ABP channel (channel 2)
    abp_all = signals[:, 2, :]

    # SBP/DBP
    sbp_per_seg = np.squeeze(subset['SBP'][:])
    dbp_per_seg = np.squeeze(subset['DBP'][:])

    # Use the minimum available segments
    num_segments = min(num_segments_raw, len(sbp_per_seg))
    abp_all = abp_all[:num_segments]
    sbp_per_seg = sbp_per_seg[:num_segments]
    dbp_per_seg = dbp_per_seg[:num_segments]


    # Demographics helper
    def get_per_segment(field):
        raw = subset[field][:]
        if raw.shape[0] == 1:
            return np.full(num_segments, float(np.mean(raw)))
        else:
            return np.squeeze(raw)[:num_segments]


    age_per_seg = get_per_segment('Age')
    bmi_per_seg = get_per_segment('BMI')
    height_per_seg = get_per_segment('Height')
    weight_per_seg = get_per_segment('Weight')

    # Gender
    gender_raw = subset['Gender'][:]
    if gender_raw.shape[0] == 1:
        g = gender_raw[0]
        if isinstance(g, np.ndarray):
            g = g.flat[0]
        gender_numeric = 1 if g == b'M' else 0
        gender_per_seg = np.full(num_segments, gender_numeric)
    else:
        gender_per_seg = np.zeros(num_segments)
        for i in range(num_segments):
            g = gender_raw[i]
            if isinstance(g, np.ndarray):
                g = g.flat[0]
            gender_per_seg[i] = 1 if g == b'M' else 0

    # Stack demographics
    demographics_per_seg = np.column_stack(
        (age_per_seg, bmi_per_seg, gender_per_seg, height_per_seg, weight_per_seg)
    )

    # Pick a single subject (all segments with same demographics)
    subject_key = tuple(demographics_per_seg[0])
    subject_indices = [i for i in range(num_segments) if tuple(demographics_per_seg[i]) == subject_key]

    # Select exactly 1,000 segments if available
    num_to_select = min(1000, len(subject_indices))
    selected_indices = random.sample(subject_indices, num_to_select)

    for i in selected_indices:
        all_abp_segments.append(abp_all[i][:625])  # trim to 625 samples if needed
        all_demographics.append(demographics_per_seg[i])

print(f"Loaded 1 subject with {len(all_abp_segments)} ABP segments.")
print(f"Demographics shape: {np.array(all_demographics).shape}")

# 2. Run clustering
dc = DivisiveClustering(all_abp_segments, min_cluster_size=5, max_depth=10, dtw_window=20)
clusters = dc.run()
print(f"Total clusters formed: {len(clusters)}")

# 3. Plot cluster representatives
plot_cluster_representatives(all_abp_segments, clusters, show_n=3)

# 4. Closest pair + Kadane
for ci, cluster in enumerate(clusters):
    if len(cluster) < 2:
        continue
    subseries = [all_abp_segments[i] for i in cluster]
    (i, j), dist = closest_pair_indices(subseries, window=20)
    print(f"Cluster {ci} size {len(cluster)}: closest pair indices {i},{j}, dist {dist:.3f}")

    # Kadane on first two series
    for rel_idx, idx in enumerate(cluster[:2]):
        s = all_abp_segments[idx]
        activity = np.abs(np.diff(s))
        st, ed, val = kadane(activity)
        print(f"  series {idx}: Kadane interval [{st},{ed}] sum={val:.3f}")
