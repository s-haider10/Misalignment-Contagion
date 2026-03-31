| Model                 | Condition      | Topology   |   N agents |   EV Shift | EV CR (≥0.5)   |   DTW ρ |    ΔCC |   Median II |   Median SRF |
|:----------------------|:---------------|:-----------|-----------:|-----------:|:---------------|--------:|-------:|------------:|-------------:|
| Llama-3.1-8B-Instruct | model-induced  | Chain      |        400 |     -0.189 | 19.0%          |   2.022 | -1.222 |       0.42  |        0.737 |
| Llama-3.1-8B-Instruct | model-induced  | Circle     |        400 |     -0.08  | 25.5%          |   1.882 | -1.123 |       0.509 |        0.619 |
| Llama-3.1-8B-Instruct | model-induced  | FC         |        400 |      0.246 | 31.2%          |   0.81  | -0.608 |       1.352 |        0.275 |
| Llama-3.1-8B-Instruct | model-induced  | Star       |        400 |      0.096 | 27.8%          |   1.929 | -1.115 |       0.559 |        0.802 |
| Llama-3.1-8B-Instruct | prompt-induced | Chain      |        400 |     -0.154 | 19.0%          |   1.949 | -1.17  |       0.447 |        0.693 |
| Llama-3.1-8B-Instruct | prompt-induced | Circle     |        400 |      0.019 | 26.2%          |   1.836 | -1.105 |       0.459 |        0.724 |
| Llama-3.1-8B-Instruct | prompt-induced | FC         |        400 |      0.147 | 31.5%          |   0.751 | -0.26  |       1.373 |        0.306 |
| Llama-3.1-8B-Instruct | prompt-induced | Star       |        400 |      0.063 | 31.8%          |   1.927 | -0.9   |       0.611 |        0.725 |
| Qwen-0.5B-Instruct    | model-induced  | Chain      |        400 |      2.025 | 95.5%          |   0.678 |  0.01  |       2.905 |       -1.483 |
| Qwen-0.5B-Instruct    | model-induced  | Circle     |        400 |      1.915 | 94.0%          |   0.845 | -0.015 |       1.741 |       -0.791 |
| Qwen-0.5B-Instruct    | model-induced  | FC         |        392 |      1.777 | 89.8%          |   1.043 | -0.3   |       1.716 |       -0.788 |
| Qwen-0.5B-Instruct    | model-induced  | Star       |        400 |      1.795 | 94.5%          |   0.732 | -0.117 |       2.483 |       -1.401 |
| Qwen-7B-Base          | model-induced  | Chain      |        346 |      1.084 | 94.6%          |   1.612 | -0.703 |       1.142 |       -0.124 |
| Qwen-7B-Base          | model-induced  | Circle     |        292 |      1.131 | 92.9%          |   1.409 | -0.52  |       0.884 |       -0.009 |
| Qwen-7B-Base          | model-induced  | FC         |         50 |      0.973 | 84.6%          |   0.989 | -0.234 |       0.765 |        0.109 |
| Qwen-7B-Base          | model-induced  | Star       |        189 |      1.188 | 95.3%          |   1.339 | -0.411 |       0.987 |       -0.003 |
| Qwen-7B-Instruct      | model-induced  | Chain      |       1200 |      1.561 | 94.9%          |   1.427 | -0.054 |       0.991 |       -0.024 |
| Qwen-7B-Instruct      | model-induced  | Circle     |       1200 |      1.602 | 94.2%          |   1.268 |  0.162 |       0.827 |        0.094 |
| Qwen-7B-Instruct      | model-induced  | FC         |       1200 |      1.612 | 93.3%          |   0.896 |  0.425 |       0.687 |        0.251 |
| Qwen-7B-Instruct      | model-induced  | Star       |       1200 |      1.603 | 95.6%          |   1.192 |  0.193 |       0.894 |        0.067 |
| Qwen-7B-Instruct      | prompt-induced | Chain      |        400 |      1.539 | 94.0%          |   1.19  | -0.035 |       0.96  |        0.003 |
| Qwen-7B-Instruct      | prompt-induced | Circle     |        400 |      1.619 | 95.2%          |   1.084 |  0.105 |       0.861 |        0.061 |
| Qwen-7B-Instruct      | prompt-induced | FC         |        400 |      1.695 | 95.2%          |   0.621 |  0.455 |       0.683 |        0.237 |
| Qwen-7B-Instruct      | prompt-induced | Star       |        400 |      1.641 | 92.5%          |   0.898 |  0.087 |       0.854 |        0.074 |
