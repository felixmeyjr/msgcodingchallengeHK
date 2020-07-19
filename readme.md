# get-in-it MSG-Coding Challenge

Solution to the MSG Coding Challenge: travelling salesmen problem by Felix Meyer and Yannik Strieben (studens of Uni Hannover)

Lösung zu der MSG Coding Challenge: Travelling Salesmen Problem von Felix Meyer und Yannik Strieben 
(Studenten der Leibniz Universität Hannover)

## Solution
Total distance: 2337.22km

Order of points: [0, 11, 15, 19, 18, 3, 20, 7, 12, 5, 6, 14, 13, 17, 9, 10, 2, 1, 8, 4, 16, 0]

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) or [conda](https://www.anaconda.com/products/individual) 
to install numpy, geopy and matplotlib.

Um die notwendigen Librarys zu benutzen, kann [pip](https://pip.pypa.io/en/stable/) oder
 [conda](https://www.anaconda.com/products/individual) genutzt werden. 

```bash
pip install numpy
pip install geopy
pip install matplotlib
```

## Algorithm
We used a Held Karp Algorithm. This Held-Karp algorithm implementation takes 82s to run. 
Therefore a more faster solution was needed: 
We decided to use a K-means Clustering,
 get closests points of each cluster to each cluster, and calculates min. distances in the clusters.
That didnt work very well, so back to HK-Algorithm.

Wir haben einen Held-Karp Algorithmus gewählt. Dieser Algorithmus dauert 82s zum Ausführen. Daher wurde eine schnellere 
Lösung gesucht: Ursprünglich war ein "exotischer" Ansatz gedacht: k-means auf die Punkte, die naheliegensten Cluster optimal verbinden 
und dann mit den Start- und Endpunkten im Cluster den kürzesten Weg zwischen alle Städten im Cluster finden.
Das hat leider nicht so funktioniert, wie wir uns das gedacht haben. Daher haben wir auf den Held-Karp-Algo. zurückgegriffen.

## Usage
Simply run the heldkarp.py with the interpreter of your choice.

Einfach die heldkarp.py ausführen.


