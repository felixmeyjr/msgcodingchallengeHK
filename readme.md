# get-in-it MSG-Coding Challenge

Solution to the MSG Coding Challenge: travelling salesmen problem by Felix Meyer and Yannik Strieben

## Solution
Total distance: 2337.22km

Order of points: [0, 11, 15, 19, 18, 3, 20, 7, 12, 5, 6, 14, 13, 17, 9, 10, 2, 1, 8, 4, 16, 0]

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) or [conda](https://www.anaconda.com/products/individual) to install numpy, geopy and matplotlib.

```bash
pip install numpy
pip install geopy
pip install matplotlib
```

## Algorithm
The Held-Karp algorithm took 82s to run. Therefore a more faster solution was needed: We decided to use a K-means Clustering with k = 5, then nearest neighbor algorithm in the clusters.



