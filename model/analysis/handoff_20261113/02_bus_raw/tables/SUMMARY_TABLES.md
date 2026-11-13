## night/evening games, sum of positive excess 16-23h

```
            bus_excess  subway_excess  ratio_subway_per_bus
date                                                       
2026-05-30       223.0         2937.0                 13.17
2026-06-02       433.5         4342.5                 10.02
2026-06-03       423.0         5422.5                 12.82
2026-06-04       304.0         2646.0                  8.70
2026-06-05       381.0         4238.0                 11.12
2026-06-06       406.0         4751.0                 11.70
2026-06-07       336.0         3686.0                 10.97
2026-06-13       169.0         1469.0                  8.69
```

## day games (14:00 start), sum of positive excess 15-19h

```
            bus_excess  subway_excess  ratio_subway_per_bus
date                                                       
2026-03-28       404.0         7229.0                 17.89
2026-03-29       371.0         6689.0                 18.03
2026-04-26       404.0         5087.0                 12.59
2026-05-02       468.0         6930.0                 14.81
2026-05-03       358.0         6469.0                 18.07
2026-05-17       473.0         6449.0                 13.63
2026-06-06       383.5         4586.5                 11.96
```

## Munhak complex stops: mean ride 20-23h / 16-18h on event days vs ordinary-day median

```
hour                                        evening  afternoon  evening_ordinary_median  afternoon_ordinary_median
event                  ars   stop_name                                                                            
PLAVE(main)            37144 문학경기장(야구장)       664.0      116.0                     10.0                       35.0
                       37157 문학경기장(월드컵경기장)     72.0       18.0                     11.5                       24.0
                       38310 문학경기장(박태환수영장)      0.0        0.0                     23.5                       43.0
                       38311 문학경기장(박태환수영장)     31.0        4.0                      1.5                        3.0
                       38329 문학경기장(박태환수영장)     26.0       10.0                      4.0                       15.0
                       38423 문학경기장(박태환수영장)      4.0        8.0                      3.0                       12.0
Sustainable Wave(main) 37144 문학경기장(야구장)       182.0       40.0                     10.0                       35.0
                       37157 문학경기장(월드컵경기장)      6.0        4.0                     11.5                       24.0
baseball(ballpark)     37144 문학경기장(야구장)       245.0       90.0                     10.0                       35.0
                       37157 문학경기장(월드컵경기장)    114.0       42.0                     11.5                       24.0
                       38310 문학경기장(박태환수영장)     54.0       27.0                     23.5                       43.0
                       38311 문학경기장(박태환수영장)     27.0        7.0                      1.5                        3.0
                       38329 문학경기장(박태환수영장)     26.0       14.0                      4.0                       15.0
                       38423 문학경기장(박태환수영장)      4.0       10.0                      3.0                       12.0
concert(main)          37144 문학경기장(야구장)       528.0       46.0                     10.0                       35.0
                       37157 문학경기장(월드컵경기장)     97.0       30.0                     11.5                       24.0
                       38310 문학경기장(박태환수영장)     56.0       22.0                     23.5                       43.0
                       38311 문학경기장(박태환수영장)     30.0        6.0                      1.5                        3.0
                       38329 문학경기장(박태환수영장)     15.0        8.0                      4.0                       15.0
                       38423 문학경기장(박태환수영장)     11.0        6.0                      3.0                       12.0
day game(ballpark)     37144 문학경기장(야구장)         8.0      388.0                     10.0                       35.0
fan match(ballpark)    37144 문학경기장(야구장)       531.0       45.0                     10.0                       35.0
                       37157 문학경기장(월드컵경기장)    152.0       18.0                     11.5                       24.0
```

## Geomdan stops, weekday mean relative to June

```
         ride_vs_june  alight_vs_june  am_06_09_vs_june
month                                                  
2026-06         1.000           1.000             1.000
2026-07         0.912           0.934             0.930
2026-08         0.555           0.624             0.733
2026-09         0.528           0.443             0.651
2026-10         1.354           1.472             2.337
```

## Summer leave: weekday mean per stop

```
                                  ride                                                         alight                                        
period          leave week 1 (7/27-31) leave week 2 (8/3-7) mid-July (7/13-24) leave week 1 (7/27-31) leave week 2 (8/3-7) mid-July (7/13-24)
ars   stop_name                                                                                                                              
37583 주안국가산단역                    299.0                255.0              292.0                   26.0                 20.0               26.0
37584 주안국가산단역                      2.0                  4.0                3.0                   89.0                 89.0               88.0
```

## Same stations, subway 2025 (training) weekday day totals

```
period      leave week 1 2025  leave week 2 2025  mid-July 2025
station_id                                                     
3118                   5841.0             6182.0         6581.0
3217                   2200.0             2229.0         2878.0
3761                   6738.0             6850.0         7335.0
```

## Songdo 3139: bus excess (38696+38697) vs subway excess, 20-23h

```
      date weekday     event  bus_excess_20_23  subway_excess_20_23  ratio  v20e_20_23  bus_21  bus_22  subway_22  v20e_22
2025-08-01     Fri Pentaport             197.0               9594.0   48.7         NaN     117     143     4069.0      NaN
2025-08-02     Sat Pentaport             218.0               9939.0   45.6         NaN      98     101     4143.0      NaN
2025-08-03     Sun Pentaport             186.0               7421.0   39.9         NaN     128      86     2654.0      NaN
2025-08-22     Fri      Beer              48.5               1728.0   35.6         NaN      49     100     1210.0      NaN
2025-08-23     Sat      Beer              84.0               3568.0   42.5         NaN      74     103     2602.0      NaN
2025-08-24     Sun      Beer              44.0               2359.0   53.6         NaN      35      80     1932.0      NaN
2025-08-25     Mon      Beer              23.0                542.0   23.6         NaN      62      78      366.0      NaN
2025-08-26     Tue      Beer              23.5               1228.0   52.3         NaN      55      85     1075.0      NaN
2025-08-27     Wed      Beer              18.5               1273.0   68.8         NaN      49      80      876.0      NaN
2025-08-28     Thu      Beer              46.0               1026.0   22.3         NaN      78      85      661.0      NaN
2025-08-29     Fri      Beer              62.0               1302.0   21.0         NaN      75     104     1016.0      NaN
2025-08-30     Sat      Beer             151.0               5709.0   37.8         NaN      94     149     4230.0      NaN
2026-07-31     Fri Pentaport             291.0                  NaN    NaN      5710.0     180     134        NaN   2240.0
2026-08-01     Sat Pentaport             229.0                  NaN    NaN      8968.0      86      71        NaN   3629.0
2026-08-02     Sun Pentaport             206.0                  NaN    NaN      6297.0     108      72        NaN   2054.0
2026-08-22     Sat      Beer              26.0                  NaN    NaN      5296.0      47      24        NaN   3495.0
2026-08-23     Sun      Beer               2.0                  NaN    NaN      2355.0      18      29        NaN   1562.0
2026-08-24     Mon      Beer               0.0                  NaN    NaN       983.0      30      23        NaN    510.0
2026-08-25     Tue      Beer               3.0                  NaN    NaN      1144.0      35      26        NaN    612.0
2026-08-26     Wed      Beer               3.0                  NaN    NaN      1100.0      33      34        NaN    583.0
2026-08-27     Thu      Beer               0.0                  NaN    NaN      1111.0      37      20        NaN    602.0
2026-08-28     Fri      Beer              28.5                  NaN    NaN      1534.0      40      50        NaN    848.0
2026-08-29     Sat      Beer              32.0                  NaN    NaN      4250.0      27      41        NaN   2752.0
2026-08-30     Sun      Beer              11.0                  NaN    NaN      2100.0      30      20        NaN   1382.0
2026-09-12     Sat    Islive              73.0                  NaN    NaN      2482.0      34      53        NaN   1124.0
```

## Bupyeong-samgeori 3122: daytime 09-18 bus (39491+39493) vs subway 2025 / v20e 2026

```
      date weekday  bus_ride_09_18  bus_alight_09_18  subway_3122_09_18  v20e_3122_09_18
2025-09-27     Sat              78                97             2513.0              NaN
2025-09-28     Sun              53                54             1612.0              NaN
2025-09-29     Mon             103                95             2145.0              NaN
2025-09-30     Tue              86                83             2115.0              NaN
2025-10-01     Wed             110               103             2128.0              NaN
2025-10-02     Thu             124                95             2393.0              NaN
2025-10-03     Fri             115               122             3281.0              NaN
2025-10-04     Sat             122               159             4192.0              NaN
2025-10-05     Sun             127               160             3956.0              NaN
2025-10-06     Mon             182               251             8866.0              NaN
2025-10-07     Tue             102               117             2938.0              NaN
2025-10-08     Wed             142               145             3066.0              NaN
2025-10-09     Thu              68                93             2191.0              NaN
2025-10-10     Fri              74                54             1368.0              NaN
2026-09-14     Mon              96                65                NaN           1896.0
2026-09-15     Tue              89                58                NaN           2014.0
2026-09-16     Wed              94                71                NaN           2002.0
2026-09-17     Thu             103                70                NaN           1952.0
2026-09-18     Fri             102                82                NaN           2116.0
2026-09-19     Sat             112               128                NaN           2033.0
2026-09-20     Sun             120               109                NaN           1899.0
2026-09-21     Mon             116               110                NaN           2005.0
2026-09-22     Tue             116               111                NaN           2033.0
2026-09-23     Wed             126               105                NaN           2097.0
2026-09-24     Thu             190               201                NaN           4102.0
2026-09-25     Fri             341               347                NaN           8504.0
2026-09-26     Sat             140               133                NaN           3023.0
2026-09-27     Sun              92                96                NaN           2441.0
```

