
### T1 (FP1/3): 37 inputs, 37 moves — 95% CI by resampling moves

| method | mean cm [95% CI] | median cm [95% CI] | <10 cm [95% CI] |
|---|---|---|---|
| rc | 7.08 [4.26, 10.21] | 1.34 [0.33, 14.60] | 62.2% [46, 78] |
| agree | 2.47 [1.30, 3.80] | 0.75 [0.33, 1.88] | 91.9% [84, 100] |
| raft_crop | 5.23 [2.52, 8.51] | 0.88 [0.46, 2.72] | 78.4% [65, 92] |
| roma_pnp | 9.00 [5.67, 12.85] | 2.51 [1.41, 14.94] | 62.2% [46, 78] |
| inerf | 14.67 [11.23, 18.39] | 13.93 [9.44, 19.42] | 37.8% [22, 54] |
| cotracker | 4.41 [2.17, 6.95] | 0.99 [0.66, 2.17] | 81.1% [68, 92] |
| dispnet_v3 | 7.03 [5.23, 9.03] | 5.64 [3.75, 8.39] | 78.4% [65, 92] |
| dispnet_v3+rc | 4.57 [2.80, 6.47] | 1.37 [0.39, 5.15] | 78.4% [65, 89] |
| agree+dispnet_v3 | 2.54 [1.35, 3.90] | 0.92 [0.36, 1.88] | 91.9% [84, 100] |
| cotracker/dispnet_v3+rc | 2.87 [1.28, 4.88] | 0.93 [0.64, 1.32] | 89.2% [78, 97] |
| cotracker→dispnet_v3+rc | 3.45 [1.97, 5.22] | 0.99 [0.64, 2.14] | 83.8% [70, 95] |

- paired mean diff cotracker/dispnet_v3+rc − cotracker: -1.54 cm, 95% CI [-3.53, -0.00]
- paired mean diff cotracker/dispnet_v3+rc − agree+dispnet_v3: +0.33 cm, 95% CI [-1.27, +2.44]
- paired mean diff cotracker/dispnet_v3+rc − agree: +0.40 cm, 95% CI [-1.22, +2.48]
- paired mean diff cotracker/dispnet_v3+rc − roma_pnp: -6.13 cm, 95% CI [-9.89, -3.00]
- paired mean diff cotracker/dispnet_v3+rc − rc: -4.21 cm, 95% CI [-7.62, -0.75]
- paired mean diff cotracker/dispnet_v3+rc − dispnet_v3+rc: -1.71 cm, 95% CI [-3.23, -0.41]
- paired mean diff agree+dispnet_v3 − agree: +0.07 cm, 95% CI [-0.09, +0.24]

### T2 all (FP1-5): 891 inputs, 172 moves — 95% CI by resampling moves

| method | mean cm [95% CI] | median cm [95% CI] | <10 cm [95% CI] |
|---|---|---|---|
| rc | 3.75 [2.81, 4.84] | 0.54 [0.47, 0.69] | 88.0% [83, 92] |
| agree | 1.81 [1.41, 2.29] | 0.48 [0.42, 0.56] | 96.2% [94, 98] |
| raft_crop | 2.40 [1.71, 3.21] | 0.38 [0.32, 0.44] | 92.9% [89, 96] |
| roma_pnp | 7.22 [5.61, 9.09] | 1.53 [1.07, 2.70] | 75.1% [69, 81] |
| inerf | 9.33 [7.81, 10.91] | 4.75 [3.84, 5.49] | 70.9% [65, 76] |
| cotracker | 2.22 [1.56, 3.02] | 0.34 [0.30, 0.39] | 92.9% [89, 96] |
| dispnet_v3 | 4.11 [3.61, 4.68] | 2.64 [2.22, 3.06] | 90.5% [87, 93] |
| dispnet_v3+rc | 2.33 [1.85, 2.87] | 0.59 [0.50, 0.70] | 94.2% [91, 96] |
| agree+dispnet_v3 | 1.59 [1.27, 2.00] | 0.49 [0.43, 0.58] | 97.1% [96, 98] |
| cotracker/dispnet_v3+rc | 1.62 [1.21, 2.10] | 0.34 [0.30, 0.38] | 95.3% [93, 97] |
| cotracker→dispnet_v3+rc | 1.91 [1.45, 2.44] | 0.36 [0.31, 0.41] | 94.2% [92, 96] |

- paired mean diff cotracker/dispnet_v3+rc − cotracker: -0.59 cm, 95% CI [-1.04, -0.27]
- paired mean diff cotracker/dispnet_v3+rc − agree+dispnet_v3: +0.03 cm, 95% CI [-0.26, +0.34]
- paired mean diff cotracker/dispnet_v3+rc − agree: -0.19 cm, 95% CI [-0.48, +0.11]
- paired mean diff cotracker/dispnet_v3+rc − roma_pnp: -5.60 cm, 95% CI [-7.36, -4.14]
- paired mean diff cotracker/dispnet_v3+rc − rc: -2.13 cm, 95% CI [-3.01, -1.38]
- paired mean diff cotracker/dispnet_v3+rc − dispnet_v3+rc: -0.71 cm, 95% CI [-1.09, -0.37]
- paired mean diff agree+dispnet_v3 − agree: -0.22 cm, 95% CI [-0.43, -0.05]

### T2 new scenes only (FP2/4/5): 523 inputs, 105 moves — 95% CI by resampling moves

| method | mean cm [95% CI] | median cm [95% CI] | <10 cm [95% CI] |
|---|---|---|---|
| rc | 3.92 [2.72, 5.49] | 0.53 [0.43, 0.76] | 88.5% [83, 93] |
| agree | 1.88 [1.39, 2.59] | 0.47 [0.40, 0.58] | 96.4% [94, 98] |
| raft_crop | 2.61 [1.68, 3.86] | 0.40 [0.32, 0.54] | 93.3% [90, 96] |
| roma_pnp | 7.14 [5.53, 9.01] | 1.73 [1.00, 3.74] | 72.5% [65, 80] |
| inerf | 9.56 [7.59, 11.79] | 4.50 [3.19, 5.68] | 70.0% [62, 77] |
| cotracker | 2.46 [1.58, 3.57] | 0.37 [0.31, 0.44] | 93.3% [89, 96] |
| dispnet_v3 | 4.44 [3.82, 5.14] | 2.85 [2.45, 3.24] | 88.7% [84, 92] |
| dispnet_v3+rc | 2.49 [1.91, 3.25] | 0.58 [0.45, 0.74] | 93.3% [90, 96] |
| agree+dispnet_v3 | 1.61 [1.19, 2.25] | 0.48 [0.40, 0.65] | 97.1% [95, 99] |
| cotracker/dispnet_v3+rc | 1.77 [1.26, 2.42] | 0.37 [0.31, 0.43] | 95.2% [93, 97] |
| cotracker→dispnet_v3+rc | 2.14 [1.56, 2.89] | 0.39 [0.32, 0.50] | 93.1% [89, 96] |

- paired mean diff cotracker/dispnet_v3+rc − cotracker: -0.68 cm, 95% CI [-1.36, -0.20]
- paired mean diff cotracker/dispnet_v3+rc − agree+dispnet_v3: +0.16 cm, 95% CI [-0.26, +0.59]
- paired mean diff cotracker/dispnet_v3+rc − agree: -0.11 cm, 95% CI [-0.49, +0.31]
- paired mean diff cotracker/dispnet_v3+rc − roma_pnp: -5.36 cm, 95% CI [-6.88, -3.99]
- paired mean diff cotracker/dispnet_v3+rc − rc: -2.15 cm, 95% CI [-3.38, -1.13]
- paired mean diff cotracker/dispnet_v3+rc − dispnet_v3+rc: -0.71 cm, 95% CI [-1.31, -0.19]
- paired mean diff agree+dispnet_v3 − agree: -0.27 cm, 95% CI [-0.56, -0.02]
