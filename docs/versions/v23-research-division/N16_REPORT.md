# N16: directed probing

Does choosing probes by expected information beat the frozen fixed order? N16's 20%-randomized adaptive policy gains 0.056776 nats of true-mechanism log score at two added probes (95% whole-law interval 0.050350 to 0.063473), equally weighting N4d, N9, N11 and N15. All 49,152 cases, byte-identical replay and independent reconstruction verify. This is a positive constructed scheduler witness, not evidence about human questions or automatic engine admission.

The primary comparison is the exact scoring reader's true-mechanism log-score gain for P4, which chooses the greedy expected-information probe 80% of the time and a uniformly random remaining probe 20% of the time, over the frozen fixed order P1 after two **additional** probes. N4d still conditions on its original 0/2/6 observations. The primary first averages the twelve settings within each source study, then weights the four studies equally. Its direction was frozen as positive; the observed interval is entirely above zero.

| Source study | Mean gain, nats | 95% whole-law interval |
|---|---:|---:|
| N4d | 0.039108 | 0.031601 to 0.046399 |
| N9 | 0.080373 | 0.067878 to 0.092602 |
| N11 | 0.073495 | 0.061044 to 0.086270 |
| N15 | 0.034128 | 0.027452 to 0.040105 |

The full setting breakdown follows. A positive number favors P4. Settings are frozen generator coordinates, shown compactly rather than translated into new claims.

| Source | Frozen setting | Mean gain, nats | 95% whole-law interval |
|---|---|---:|---:|
| N4d | `{"mechanism":0,"probes":0}` | 0.030440 | -0.007209 to 0.066692 |
| N4d | `{"mechanism":0,"probes":2}` | 0.066425 | 0.038017 to 0.096501 |
| N4d | `{"mechanism":0,"probes":6}` | 0.042662 | 0.023281 to 0.063832 |
| N4d | `{"mechanism":1,"probes":0}` | 0.149998 | 0.126928 to 0.172407 |
| N4d | `{"mechanism":1,"probes":2}` | 0.110335 | 0.072961 to 0.146059 |
| N4d | `{"mechanism":1,"probes":6}` | 0.040766 | 0.021499 to 0.060709 |
| N4d | `{"mechanism":2,"probes":0}` | 0.075683 | 0.039646 to 0.106895 |
| N4d | `{"mechanism":2,"probes":2}` | 0.024276 | -0.007057 to 0.053468 |
| N4d | `{"mechanism":2,"probes":6}` | 0.028112 | 0.009957 to 0.046477 |
| N4d | `{"mechanism":3,"probes":0}` | -0.030294 | -0.054590 to -0.002744 |
| N4d | `{"mechanism":3,"probes":2}` | -0.040562 | -0.084779 to 0.010721 |
| N4d | `{"mechanism":3,"probes":6}` | -0.028550 | -0.046310 to -0.008613 |
| N9 | `{"history":0,"timing":false,"width":1}` | 0.098988 | 0.057207 to 0.137614 |
| N9 | `{"history":0,"timing":true,"width":1}` | 0.114486 | 0.075874 to 0.155649 |
| N9 | `{"history":8,"timing":false,"width":1}` | 0.027467 | 0.003182 to 0.052542 |
| N9 | `{"history":8,"timing":true,"width":1}` | 0.015817 | 0.001788 to 0.029563 |
| N9 | `{"history":32,"timing":false,"width":1}` | 0.018097 | -0.005038 to 0.041805 |
| N9 | `{"history":32,"timing":true,"width":1}` | -0.000715 | -0.002406 to 0.000409 |
| N9 | `{"history":0,"timing":false,"width":2}` | 0.104161 | 0.052841 to 0.149812 |
| N9 | `{"history":0,"timing":true,"width":2}` | 0.184417 | 0.116682 to 0.246786 |
| N9 | `{"history":8,"timing":false,"width":2}` | 0.106342 | 0.079929 to 0.133861 |
| N9 | `{"history":8,"timing":true,"width":2}` | 0.105478 | 0.084020 to 0.128743 |
| N9 | `{"history":32,"timing":false,"width":2}` | 0.088179 | 0.053877 to 0.124458 |
| N9 | `{"history":32,"timing":true,"width":2}` | 0.101760 | 0.077884 to 0.125351 |
| N11 | `{"history":0,"mechanism":0}` | 0.063557 | 0.017039 to 0.107294 |
| N11 | `{"history":8,"mechanism":0}` | 0.041338 | 0.009553 to 0.079959 |
| N11 | `{"history":32,"mechanism":0}` | 0.051755 | -0.001327 to 0.101056 |
| N11 | `{"history":0,"mechanism":1}` | 0.145501 | 0.105398 to 0.183066 |
| N11 | `{"history":8,"mechanism":1}` | 0.120956 | 0.069132 to 0.168433 |
| N11 | `{"history":32,"mechanism":1}` | 0.156098 | 0.105933 to 0.202735 |
| N11 | `{"history":0,"mechanism":2}` | 0.041577 | -0.013836 to 0.091510 |
| N11 | `{"history":8,"mechanism":2}` | 0.069651 | 0.046046 to 0.094203 |
| N11 | `{"history":32,"mechanism":2}` | 0.114743 | 0.076521 to 0.152215 |
| N11 | `{"history":0,"mechanism":3}` | 0.043947 | 0.014256 to 0.072154 |
| N11 | `{"history":8,"mechanism":3}` | 0.013230 | -0.016160 to 0.043996 |
| N11 | `{"history":32,"mechanism":3}` | 0.019592 | -0.003360 to 0.041501 |
| N15 | `{"history":0,"mechanism":0}` | 0.053889 | 0.028393 to 0.077565 |
| N15 | `{"history":8,"mechanism":0}` | 0.034222 | 0.010976 to 0.056726 |
| N15 | `{"history":32,"mechanism":0}` | 0.001616 | -0.004522 to 0.008049 |
| N15 | `{"history":0,"mechanism":1}` | 0.042484 | 0.017223 to 0.067385 |
| N15 | `{"history":8,"mechanism":1}` | 0.010008 | -0.005942 to 0.025353 |
| N15 | `{"history":32,"mechanism":1}` | -0.002535 | -0.007357 to 0.002602 |
| N15 | `{"history":0,"mechanism":2}` | -0.000787 | -0.017370 to 0.015794 |
| N15 | `{"history":8,"mechanism":2}` | -0.003841 | -0.020789 to 0.011414 |
| N15 | `{"history":32,"mechanism":2}` | -0.005655 | -0.015439 to 0.002280 |
| N15 | `{"history":0,"mechanism":3}` | 0.202912 | 0.163052 to 0.239870 |
| N15 | `{"history":8,"mechanism":3}` | 0.071228 | 0.049912 to 0.096924 |
| N15 | `{"history":32,"mechanism":3}` | 0.005999 | -0.001379 to 0.014083 |

P4 does not match the fixed order's six-probe gain at any shorter registered prefix: the mean P4-minus-P1-at-six comparison is negative at one, two, three and four probes, and numerical zero at six. This is expected convergence after exhausting the same six-item menu, not an equivalence test. The exact values and intervals for every prefix are in the machine secondary record.

Misspecification changes the result. P3 minus P2 measures greedy expected-information selection against random selection; negative values are active-learning harm. P5 minus P3 measures whether exponent-0.5 likelihood tempering repairs or harms the greedy policy. The table equally weights the four source studies; source-specific and all-setting values remain in the machine record.

| Policy model | P3 minus random, mean | 95% interval | Tempered P5 minus P3, mean | 95% interval |
|---|---:|---:|---:|---:|
| correct | 0.065432 | 0.055058 to 0.075445 | 0.000243 | -0.000437 to 0.000962 |
| high-reliability | 0.065401 | 0.054991 to 0.075540 | 0.000276 | -0.000515 to 0.001002 |
| omit-0 | 0.044231 | 0.035856 to 0.052152 | -0.000081 | -0.000669 to 0.000639 |
| omit-1 | 0.034365 | 0.025362 to 0.042987 | -0.000483 | -0.001668 to 0.000853 |
| omit-2 | 0.038740 | 0.031266 to 0.045958 | -0.001874 | -0.003075 to -0.000672 |
| omit-3 | 0.047716 | 0.038894 to 0.055664 | 0.001201 | 0.000099 to 0.002360 |

The source breakdown prevents one study from hiding another study's active-learning harm.

| Policy model | Source | P3 minus random, mean | 95% interval | Tempered P5 minus P3, mean | 95% interval |
|---|---|---:|---:|---:|---:|
| correct | N4d | 0.032213 | 0.020856 to 0.043128 | 0.000000 | 0.000000 to 0.000000 |
| correct | N9 | 0.099103 | 0.078738 to 0.119390 | -0.000001 | -0.000195 to 0.000170 |
| correct | N11 | 0.086862 | 0.074146 to 0.099018 | 0.000636 | -0.002092 to 0.003281 |
| correct | N15 | 0.043551 | 0.036144 to 0.050739 | 0.000339 | -0.000518 to 0.001218 |
| high-reliability | N4d | 0.032213 | 0.020443 to 0.043339 | 0.000000 | 0.000000 to 0.000000 |
| high-reliability | N9 | 0.099072 | 0.078481 to 0.118706 | 0.000076 | -0.000568 to 0.000749 |
| high-reliability | N11 | 0.086862 | 0.074100 to 0.099648 | 0.000636 | -0.001965 to 0.003281 |
| high-reliability | N15 | 0.043457 | 0.036205 to 0.051047 | 0.000393 | -0.000257 to 0.000929 |
| omit-0 | N4d | -0.023354 | -0.034370 to -0.011144 | 0.000000 | 0.000000 to 0.000000 |
| omit-0 | N9 | 0.069282 | 0.053776 to 0.084560 | -0.000061 | -0.002432 to 0.002374 |
| omit-0 | N11 | 0.086733 | 0.073802 to 0.100096 | 0.000003 | 0.000000 to 0.000010 |
| omit-0 | N15 | 0.044265 | 0.036576 to 0.052003 | -0.000267 | -0.001445 to 0.001004 |
| omit-1 | N4d | -0.014678 | -0.027528 to -0.001399 | 0.000000 | 0.000000 to 0.000000 |
| omit-1 | N9 | 0.061531 | 0.045676 to 0.077076 | 0.000451 | -0.003616 to 0.004798 |
| omit-1 | N11 | 0.046772 | 0.036798 to 0.057288 | -0.002296 | -0.003854 to -0.000491 |
| omit-1 | N15 | 0.043833 | 0.036575 to 0.051303 | -0.000088 | -0.000396 to 0.000212 |
| omit-2 | N4d | -0.025843 | -0.036445 to -0.015123 | 0.000000 | 0.000000 to 0.000000 |
| omit-2 | N9 | 0.050314 | 0.034248 to 0.067465 | -0.007827 | -0.011004 to -0.004717 |
| omit-2 | N11 | 0.086785 | 0.074105 to 0.099413 | 0.000018 | -0.002996 to 0.002990 |
| omit-2 | N15 | 0.043705 | 0.036252 to 0.050864 | 0.000312 | -0.000755 to 0.001410 |
| omit-3 | N4d | 0.032213 | 0.020770 to 0.043115 | 0.000000 | 0.000000 to 0.000000 |
| omit-3 | N9 | 0.066544 | 0.052437 to 0.080085 | 0.005118 | 0.001805 to 0.009104 |
| omit-3 | N11 | 0.087971 | 0.075255 to 0.101481 | -0.000473 | -0.004397 to 0.003732 |
| omit-3 | N15 | 0.004136 | -0.001646 to 0.010341 | 0.000157 | -0.000611 to 0.001027 |

The mechanism-omission arms also retain the true-omitted subsets separately. Those ratios use only cases whose planted mechanism is the one removed from the policy model; no unavailable case is imputed. This separation matters because an average over every case can hide the harm where the policy's model is wrong in exactly the planted way.

| Omitted policy mechanism | Source | Actual true-omitted cases | P3 minus random, mean | 95% whole-law interval |
|---|---|---:|---:|---:|
| omit-0 | N4d | 3072 | -0.130122 | -0.153174 to -0.108413 |
| omit-0 | N9 | 2460 | 0.010189 | -0.010329 to 0.029378 |
| omit-0 | N11 | 3072 | 0.032333 | 0.010487 to 0.051906 |
| omit-0 | N15 | 3072 | 0.041466 | 0.028460 to 0.053238 |
| omit-1 | N4d | 3072 | -0.016859 | -0.036314 to 0.000365 |
| omit-1 | N9 | 2460 | 0.073736 | 0.049549 to 0.098767 |
| omit-1 | N11 | 3072 | 0.024478 | -0.003472 to 0.052889 |
| omit-1 | N15 | 3072 | 0.020681 | 0.011324 to 0.029238 |
| omit-2 | N4d | 3072 | -0.132400 | -0.152046 to -0.108825 |
| omit-2 | N9 | 2460 | -0.183420 | -0.213320 to -0.157082 |
| omit-2 | N11 | 3072 | 0.098956 | 0.077571 to 0.119052 |
| omit-2 | N15 | 3072 | -0.001033 | -0.009396 to 0.006820 |
| omit-3 | N4d | 3072 | -0.054858 | -0.076276 to -0.033297 |
| omit-3 | N9 | 2460 | 0.099957 | 0.075085 to 0.122976 |
| omit-3 | N11 | 3072 | 0.025134 | 0.002761 to 0.047012 |
| omit-3 | N15 | 3072 | 0.000822 | -0.015667 to 0.018312 |

Tempering is therefore not a generic recommendation. It is routed only where the measured P5-minus-P3 comparison is positive; negative comparisons are reported as harm. A mechanism ensemble and P4's random share remain necessary protections wherever greedy selection falls below random under misspecification.

Selection propensities were retained for every step, but support is limited. P1, P3 and P5 are deterministic and assign zero probability to unchosen probes; arbitrary retrospective reweighting from their paths is unavailable. P2 and P4 have sequential positivity on the finite without-replacement menu. This batch supports only the declared paired policy comparisons, not unrestricted off-policy estimates.

The following table reports expected information for the correctly specified policy model. Totals are divided by the actual number of times each probe type was selected; missing selections have count zero rather than disappearing. The machine record additionally contains every misspecified scenario and policy.

| Source | Policy | Probe type | Actual selections | Total information, nats | Nats per selected probe |
|---|---|---|---:|---:|---:|
| N4d | P1 | `[1.0, 3.0]` | 37440 | 3400.340792 | 0.090821 |
| N4d | P1 | `[3.0, 1.0]` | 36288 | 3302.392300 | 0.091005 |
| N4d | P2 | `[1.0, 3.0]` | 37440 | 3411.382125 | 0.091116 |
| N4d | P2 | `[3.0, 1.0]` | 36288 | 3302.797585 | 0.091016 |
| N4d | P3 | `[1.0, 3.0]` | 37440 | 3405.897723 | 0.090969 |
| N4d | P3 | `[3.0, 1.0]` | 36288 | 3312.805893 | 0.091292 |
| N4d | P4 | `[1.0, 3.0]` | 37440 | 3402.563523 | 0.090880 |
| N4d | P4 | `[3.0, 1.0]` | 36288 | 3312.504487 | 0.091284 |
| N4d | P5 | `[1.0, 3.0]` | 37440 | 3405.897723 | 0.090969 |
| N4d | P5 | `[3.0, 1.0]` | 36288 | 3312.805893 | 0.091292 |
| N9 | P1 | `[0, 0]` | 12288 | 1118.271868 | 0.091005 |
| N9 | P1 | `[0, 1]` | 12288 | 752.920511 | 0.061273 |
| N9 | P1 | `[1, 0]` | 12288 | 1657.045176 | 0.134851 |
| N9 | P1 | `[1, 1]` | 12288 | 699.383094 | 0.056916 |
| N9 | P1 | `[2, 0]` | 12288 | 947.834481 | 0.077135 |
| N9 | P1 | `[2, 1]` | 12288 | 694.528394 | 0.056521 |
| N9 | P2 | `[0, 0]` | 12288 | 1117.158043 | 0.090915 |
| N9 | P2 | `[0, 1]` | 12288 | 695.630517 | 0.056611 |
| N9 | P2 | `[1, 0]` | 12288 | 1665.258119 | 0.135519 |
| N9 | P2 | `[1, 1]` | 12288 | 710.354754 | 0.057809 |
| N9 | P2 | `[2, 0]` | 12288 | 962.020776 | 0.078289 |
| N9 | P2 | `[2, 1]` | 12288 | 732.630794 | 0.059622 |
| N9 | P3 | `[0, 0]` | 12288 | 1400.508250 | 0.113974 |
| N9 | P3 | `[0, 1]` | 12288 | 710.844953 | 0.057849 |
| N9 | P3 | `[1, 0]` | 12288 | 1688.164582 | 0.137383 |
| N9 | P3 | `[1, 1]` | 12288 | 679.819391 | 0.055324 |
| N9 | P3 | `[2, 0]` | 12288 | 707.983773 | 0.057616 |
| N9 | P3 | `[2, 1]` | 12288 | 684.078622 | 0.055670 |
| N9 | P4 | `[0, 0]` | 12288 | 1352.117837 | 0.110036 |
| N9 | P4 | `[0, 1]` | 12288 | 714.309969 | 0.058131 |
| N9 | P4 | `[1, 0]` | 12288 | 1675.306560 | 0.136337 |
| N9 | P4 | `[1, 1]` | 12288 | 685.995110 | 0.055826 |
| N9 | P4 | `[2, 0]` | 12288 | 743.510480 | 0.060507 |
| N9 | P4 | `[2, 1]` | 12288 | 693.019173 | 0.056398 |
| N9 | P5 | `[0, 0]` | 12288 | 1401.684070 | 0.114069 |
| N9 | P5 | `[0, 1]` | 12288 | 710.961539 | 0.057858 |
| N9 | P5 | `[1, 0]` | 12288 | 1689.689293 | 0.137507 |
| N9 | P5 | `[1, 1]` | 12288 | 679.809206 | 0.055323 |
| N9 | P5 | `[2, 0]` | 12288 | 708.190496 | 0.057633 |
| N9 | P5 | `[2, 1]` | 12288 | 682.033915 | 0.055504 |
| N11 | P1 | `[0, 0.1, 0]` | 24576 | 541.516518 | 0.022034 |
| N11 | P1 | `[1, 0.5, 2]` | 24576 | 2325.418796 | 0.094622 |
| N11 | P1 | `[1, 1.0, 1]` | 24576 | 3013.481918 | 0.122619 |
| N11 | P2 | `[0, 0.1, 0]` | 24576 | 542.435788 | 0.022072 |
| N11 | P2 | `[1, 0.5, 2]` | 24576 | 2330.038936 | 0.094810 |
| N11 | P2 | `[1, 1.0, 1]` | 24576 | 3007.729706 | 0.122385 |
| N11 | P3 | `[0, 0.1, 0]` | 24576 | 351.906112 | 0.014319 |
| N11 | P3 | `[1, 0.5, 2]` | 24576 | 1783.044443 | 0.072552 |
| N11 | P3 | `[1, 1.0, 1]` | 24576 | 3742.244060 | 0.152272 |
| N11 | P4 | `[0, 0.1, 0]` | 24576 | 391.784018 | 0.015942 |
| N11 | P4 | `[1, 0.5, 2]` | 24576 | 1842.360759 | 0.074966 |
| N11 | P4 | `[1, 1.0, 1]` | 24576 | 3636.712620 | 0.147978 |
| N11 | P5 | `[0, 0.1, 0]` | 24576 | 350.788838 | 0.014274 |
| N11 | P5 | `[1, 0.5, 2]` | 24576 | 1749.596427 | 0.071191 |
| N11 | P5 | `[1, 1.0, 1]` | 24576 | 3775.823721 | 0.153639 |
| N15 | P1 | `[0, 0]` | 24576 | 215.165079 | 0.008755 |
| N15 | P1 | `[0, 1]` | 12288 | 484.940833 | 0.039465 |
| N15 | P1 | `[1, 0]` | 24576 | 210.015063 | 0.008546 |
| N15 | P1 | `[1, 1]` | 12288 | 627.558234 | 0.051071 |
| N15 | P2 | `[0, 0]` | 24576 | 215.969739 | 0.008788 |
| N15 | P2 | `[0, 1]` | 12288 | 486.659820 | 0.039604 |
| N15 | P2 | `[1, 0]` | 24576 | 209.936966 | 0.008542 |
| N15 | P2 | `[1, 1]` | 12288 | 630.999016 | 0.051351 |
| N15 | P3 | `[0, 0]` | 24576 | 199.935213 | 0.008135 |
| N15 | P3 | `[0, 1]` | 12288 | 448.262595 | 0.036480 |
| N15 | P3 | `[1, 0]` | 24576 | 198.600268 | 0.008081 |
| N15 | P3 | `[1, 1]` | 12288 | 695.070331 | 0.056565 |
| N15 | P4 | `[0, 0]` | 24576 | 200.690351 | 0.008166 |
| N15 | P4 | `[0, 1]` | 12288 | 449.861573 | 0.036610 |
| N15 | P4 | `[1, 0]` | 24576 | 200.649556 | 0.008164 |
| N15 | P4 | `[1, 1]` | 12288 | 690.612772 | 0.056202 |
| N15 | P5 | `[0, 0]` | 24576 | 199.317696 | 0.008110 |
| N15 | P5 | `[0, 1]` | 12288 | 448.314670 | 0.036484 |
| N15 | P5 | `[1, 0]` | 24576 | 198.582142 | 0.008080 |
| N15 | P5 | `[1, 1]` | 12288 | 695.655697 | 0.056613 |

Controls pass for the uniquely informative first choice, zero information under complete aliasing, probability-product/log-posterior agreement, normalized finite selection distributions, exclusion of N9's hidden timing bit and preservation of scoring support when the policy model omits a mechanism. Original and replay raw streams match byte-for-byte. Independent review rebuilt all 49,152 scores, all canonical ledgers and every whole-law interval. The family charged 9513.109375 CPU-seconds including its independent review, inside the unchanged twelve-hour cap; no cells were shed and no V23 confirmation was run.

Route: **the positive primary passes the constructed scheduler gate.** The engine scheduler and owner probe card may carry this as a V24 test witness with the measured 20% random share and per-type values. It does not itself admit an engine module. V24 production remains prohibited before October 26 at 08:00 Pacific, and any powered confirmation remains a separate V24 study.

[Protocol](N16_PROTOCOL.md), [machine summary](../../../results/v23/n16/SUMMARY.json), [full secondary comparisons](../../../results/v23/n16/SECONDARY.json), [proof](../../../results/v23/n16/PROOF.json), [raw hashes](../../../results/v23/n16/RAW_INDEX.json).
