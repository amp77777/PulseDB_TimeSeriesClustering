# PulseDB Time-Series Clustering Project

## Project Overview

This project implements an **unsupervised clustering system for physiological time-series data** extracted from the PulseDB dataset. Using **divide-and-conquer strategies** and algorithmic reasoning, the system groups similar segments of arterial blood pressure (ABP) signals, identifies the closest pair of time-series within clusters, and analyzes peak activity intervals using **Kadane's algorithm**.

Unlike machine learning approaches, this project emphasizes **algorithmic design** to interpret biomedical signals, making it easier to understand why certain segments cluster together.

---

## Problem Description

The goal is to group 10-second ABP segments (1,000 segments from a single subject) based on similarity. After clustering:

1. Identify the **most similar pair** of time series within each cluster to validate cluster cohesion.
2. Apply **Kadane’s algorithm** to detect intervals of maximum cumulative change in each segment.
3. Generate visualizations to summarize cluster structure, representative signals, and peak activity intervals.

This approach demonstrates that **classic algorithms** (divide-and-conquer, closest pair, Kadane) can extract meaningful insights from biomedical time-series data.

---

## Dataset

- **Source:** [PulseDB VitalDB subset](https://github.com/pulselabteam/PulseDB)
- **Segments:** 1,000 ABP segments
- **Segment Length:** 10 seconds (~625 samples)
- **Demographics:** Age, BMI, Gender, Height, Weight

---

## Code Structure

- `PulseDB.py`: Handles **loading and preprocessing** the PulseDB dataset, including ABP signals and demographics.
- `pulseDB_clustering.py`: Performs **divide-and-conquer clustering**, identifies closest pairs within clusters, and applies Kadane’s algorithm to segments.
- `ts_cluster_divide_and_conquer.py`: Contains core clustering algorithms:
  - `DivisiveClustering` — recursive clustering using DTW distance.
  - `closest_pair_indices` — identifies closest pair within clusters.
  - `kadane` — maximum subarray algorithm for peak activity detection.
  - `plot_cluster_representatives` — visualizes representative signals per cluster.

---

## Usage

1. **Install dependencies:**

```bash
pip install numpy matplotlib h5py

## Block diagram
                   ┌─────────────────────┐
                   │   PulseDB Dataset   │
                   │  (1000 ABP segments)│
                   └─────────┬───────────┘
                             │
                             ▼
                   ┌─────────────────────┐
                   │ Preprocessing       │
                   │ - Extract ABP       │
                   │ - Demographics      │
                   │ - Segment trimming  │
                   └─────────┬───────────┘
                             │
                             ▼
                   ┌─────────────────────┐
                   │ Divide-and-Conquer  │
                   │ Clustering           │
                   │ - DTW similarity    │
                   │ - Recursive splitting│
                   └─────────┬───────────┘
                             │
                             ▼
            ┌────────────────────────────────┐
            │ Cluster Analysis                │
            │                                │
            │  ┌─────────────┐  ┌─────────┐ │
            │  │ Closest Pair│  │ Kadane  │ │
            │  │ Algorithm   │  │ Algorithm│ │
            │  │ (within     │  │ (max    │ │
            │  │ cluster)    │  │ activity)│ │
            │  └─────────────┘  └─────────┘ │
            └───────────────┬────────────────┘
                            │
                            ▼
                   ┌─────────────────────┐
                   │ Visualization &     │
                   │ Reporting           │
                   │ - Cluster plots     │
                   │ - Closest pair info │
                   │ - Activity windows  │
                   └─────────────────────┘
