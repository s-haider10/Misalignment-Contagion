| Dataset       | Topology   | Prompt Strategy   |   N agents |   EV Shift | EV CR (≥0.5)   |   DTW ρ |    ΔCC |   Median II |   Median SRF |
|:--------------|:-----------|:------------------|-----------:|-----------:|:---------------|--------:|-------:|------------:|-------------:|
| HarmBench-Std | FC         | Lenient:Lenient   |       1600 |      3.176 | 87.4%          |   0.266 |  1.288 |       0.932 |        0.217 |
| HarmBench-Std | FC         | Lenient:Rigid     |       1600 |      2.607 | 82.8%          |   0.744 |  0.818 |       0.986 |        0.274 |
| HarmBench-Std | FC         | Rigid:Lenient     |       1600 |      3.368 | 90.0%          |   0.364 |  0.922 |       0.992 |        0.096 |
| HarmBench-Std | FC         | Rigid:Rigid       |       1600 |      2.872 | 86.7%          |   0.831 |  0.273 |       1.005 |        0.138 |
| HarmBench-Std | Star       | Lenient:Lenient   |       1600 |      3.855 | 90.7%          |   0.653 |  0.085 |       1     |        0     |
| HarmBench-Std | Star       | Lenient:Rigid     |       1600 |      3.151 | 86.6%          |   1.081 | -0.615 |       1.083 |        0.005 |
| HarmBench-Std | Star       | Rigid:Lenient     |       1600 |      3.436 | 86.9%          |   0.897 | -0.237 |       1.019 |       -0     |
| HarmBench-Std | Star       | Rigid:Rigid       |       1600 |      2.868 | 84.4%          |   1.411 | -1.091 |       1.386 |       -0     |
| Synthetic     | FC         | Lenient:Lenient   |        400 |      1.246 | 87.5%          |   0.557 |  0.525 |       0.579 |        0.304 |
| Synthetic     | FC         | Lenient:Rigid     |        400 |      1.207 | 87.2%          |   0.573 |  0.527 |       0.585 |        0.319 |
| Synthetic     | FC         | Rigid:Lenient     |        400 |      1.622 | 95.8%          |   0.677 |  0.492 |       0.693 |        0.266 |
| Synthetic     | FC         | Rigid:Rigid       |        400 |      1.597 | 95.0%          |   0.738 |  0.448 |       0.693 |        0.266 |
| Synthetic     | Star       | Lenient:Lenient   |        400 |      1.203 | 80.2%          |   0.784 |  0.28  |       0.767 |        0.158 |
| Synthetic     | Star       | Lenient:Rigid     |        400 |      1.208 | 83.0%          |   0.805 |  0.29  |       0.779 |        0.231 |
| Synthetic     | Star       | Rigid:Lenient     |        400 |      1.563 | 93.5%          |   1.036 |  0.163 |       0.915 |        0.088 |
| Synthetic     | Star       | Rigid:Rigid       |        400 |      1.588 | 92.5%          |   1.048 |  0.107 |       0.927 |        0.071 |
