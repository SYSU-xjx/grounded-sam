# Attribute Eval Analysis

## Overall

- num_tasks: 1854
- num_hit: 617
- acc: 0.3328
- mean_iou_all: 0.2891

## Image-level Stats

- num_images: 179
- mean_image_acc: 0.3439
- p25_image_acc: 0.1667
- p50_image_acc: 0.3333
- p75_image_acc: 0.5
- min_image_acc: 0.0
- max_image_acc: 1.0

## By Attribute

| attribute_type | num_tasks | num_hit | acc | mean_iou |
| --- | --- | --- | --- | --- |
| size | 618 | 232 | 0.3754 | 0.3288 |
| spatial | 1236 | 385 | 0.3115 | 0.2693 |

## By Category

| category_en | num_tasks | num_hit | acc | mean_iou |
| --- | --- | --- | --- | --- |
| person | 582 | 123 | 0.2113 | 0.1941 |
| chair | 240 | 86 | 0.3583 | 0.3073 |
| bottle | 180 | 80 | 0.4444 | 0.3812 |
| car | 174 | 52 | 0.2989 | 0.2616 |
| cup | 174 | 47 | 0.2701 | 0.2609 |
| backpack | 132 | 48 | 0.3636 | 0.2599 |
| bicycle | 132 | 45 | 0.3409 | 0.2679 |
| laptop | 120 | 60 | 0.5 | 0.448 |
| dog | 114 | 70 | 0.614 | 0.5351 |
| microwave | 6 | 6 | 1.0 | 0.8836 |

## Miss Reason Distribution

| miss_reason | count | ratio_in_miss | ratio_in_all_tasks |
| --- | --- | --- | --- |
| low_iou | 517 | 0.4179 | 0.2789 |
| wrong_instance | 488 | 0.3945 | 0.2632 |
| no_box | 232 | 0.1876 | 0.1251 |

## Top-20 Low-IoU Tasks

| task_id | image_id | file_name | attribute_type | category_en | prompt_zh | best_iou | best_pred_score | miss_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 138550_374914_spatial_rightmost | 138550 | 000000138550.jpg | spatial | chair | 最右侧的椅子 | 0.2992 | 0.7581 | low_iou |
| 138550_374914_spatial_topmost | 138550 | 000000138550.jpg | spatial | chair | 最上方的椅子 | 0.2988 | 0.7512 | low_iou |
| 568981_499140_spatial_bottommost | 568981 | 000000568981.jpg | spatial | person | 最下方的人 | 0.2987 | 0.6553 | low_iou |
| 138550_374914_size_smallest | 138550 | 000000138550.jpg | size | chair | 最小的椅子 | 0.2946 | 0.767 | low_iou |
| 397354_1882278_size_smallest | 397354 | 000000397354.jpg | size | cup | 最小的杯子 | 0.2942 | 0.5802 | low_iou |
| 14439_375587_spatial_bottommost | 14439 | 000000014439.jpg | spatial | chair | 最下方的椅子 | 0.2911 | 0.6642 | low_iou |
| 38829_128391_spatial_bottommost | 38829 | 000000038829.jpg | spatial | bicycle | 最下方的自行车 | 0.2903 | 0.8138 | low_iou |
| 38829_128391_spatial_rightmost | 38829 | 000000038829.jpg | spatial | bicycle | 最右侧的自行车 | 0.2892 | 0.869 | low_iou |
| 356248_542388_size_smallest | 356248 | 000000356248.jpg | size | person | 最小的人 | 0.2881 | 0.6062 | low_iou |
| 448076_198115_spatial_bottommost | 448076 | 000000448076.jpg | spatial | person | 最下方的人 | 0.288 | 0.6132 | low_iou |
| 347335_673302_spatial_bottommost | 347335 | 000000347335.jpg | spatial | cup | 最下方的杯子 | 0.2868 | 0.6816 | low_iou |
| 38829_128391_size_smallest | 38829 | 000000038829.jpg | size | bicycle | 最小的自行车 | 0.2867 | 0.8818 | low_iou |
| 410880_2223427_size_largest | 410880 | 000000410880.jpg | size | chair | 最大的椅子 | 0.2863 | 0.5875 | low_iou |
| 198641_2134578_size_smallest | 198641 | 000000198641.jpg | size | laptop | 最小的笔记本电脑 | 0.2832 | 0.9144 | low_iou |
| 198641_2134578_spatial_rightmost | 198641 | 000000198641.jpg | spatial | laptop | 最右侧的笔记本电脑 | 0.2831 | 0.9391 | low_iou |
| 94944_1825525_spatial_rightmost | 94944 | 000000094944.jpg | spatial | backpack | 最右侧的背包 | 0.2822 | 0.6161 | low_iou |
| 94944_1825525_size_largest | 94944 | 000000094944.jpg | size | backpack | 最大的背包 | 0.2822 | 0.6447 | low_iou |
| 94944_1825525_spatial_bottommost | 94944 | 000000094944.jpg | spatial | backpack | 最下方的背包 | 0.2815 | 0.614 | low_iou |
| 14439_375587_size_smallest | 14439 | 000000014439.jpg | size | chair | 最小的椅子 | 0.2804 | 0.6422 | low_iou |
| 198641_2134578_spatial_bottommost | 198641 | 000000198641.jpg | spatial | laptop | 最下方的笔记本电脑 | 0.2789 | 0.8908 | low_iou |

## Top-20 Wrong-Instance Tasks

| task_id | image_id | file_name | attribute_type | category_en | prompt_zh | best_iou | best_pred_score | miss_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 183716_455359_size_largest | 183716 | 000000183716.jpg | size | person | 最大的人 | 0.1973 | 0.8355 | wrong_instance |
| 520659_1931072_spatial_leftmost | 520659 | 000000520659.jpg | spatial | chair | 最左侧的椅子 | 0.197 | 0.6512 | wrong_instance |
| 242060_1501568_spatial_topmost | 242060 | 000000242060.jpg | spatial | cup | 最上方的杯子 | 0.1859 | 0.5829 | wrong_instance |
| 84650_1823740_size_smallest | 84650 | 000000084650.jpg | size | backpack | 最小的背包 | 0.1842 | 0.5838 | wrong_instance |
| 269113_18383_spatial_rightmost | 269113 | 000000269113.jpg | spatial | dog | 最右侧的狗 | 0.1783 | 0.6144 | wrong_instance |
| 74058_344274_spatial_leftmost | 74058 | 000000074058.jpg | spatial | bicycle | 最左侧的自行车 | 0.1646 | 0.7273 | wrong_instance |
| 74058_344274_size_smallest | 74058 | 000000074058.jpg | size | bicycle | 最小的自行车 | 0.1626 | 0.768 | wrong_instance |
| 286553_445938_spatial_leftmost | 286553 | 000000286553.jpg | spatial | person | 最左侧的人 | 0.1472 | 0.6547 | wrong_instance |
| 286553_84515_spatial_leftmost | 286553 | 000000286553.jpg | spatial | bottle | 最左侧的瓶子 | 0.1403 | 0.7954 | wrong_instance |
| 286553_84515_spatial_bottommost | 286553 | 000000286553.jpg | spatial | bottle | 最下方的瓶子 | 0.14 | 0.8214 | wrong_instance |
| 286553_84515_size_smallest | 286553 | 000000286553.jpg | size | bottle | 最小的瓶子 | 0.1384 | 0.8441 | wrong_instance |
| 486040_1103890_size_smallest | 486040 | 000000486040.jpg | size | laptop | 最小的笔记本电脑 | 0.1382 | 0.7419 | wrong_instance |
| 486040_1103890_spatial_topmost | 486040 | 000000486040.jpg | spatial | laptop | 最上方的笔记本电脑 | 0.1381 | 0.7442 | wrong_instance |
| 486040_1103890_spatial_leftmost | 486040 | 000000486040.jpg | spatial | laptop | 最左侧的笔记本电脑 | 0.1376 | 0.7132 | wrong_instance |
| 397354_260673_size_largest | 397354 | 000000397354.jpg | size | person | 最大的人 | 0.1332 | 0.8634 | wrong_instance |
| 89880_6105_size_largest | 89880 | 000000089880.jpg | size | dog | 最大的狗 | 0.1268 | 0.6039 | wrong_instance |
| 411938_193398_size_largest | 411938 | 000000411938.jpg | size | person | 最大的人 | 0.1239 | 0.642 | wrong_instance |
| 210855_92275_size_largest | 210855 | 000000210855.jpg | size | bottle | 最大的瓶子 | 0.1176 | 0.6538 | wrong_instance |
| 32610_1103853_spatial_rightmost | 32610 | 000000032610.jpg | spatial | laptop | 最右侧的笔记本电脑 | 0.1162 | 0.5833 | wrong_instance |
| 256916_382283_size_largest | 256916 | 000000256916.jpg | size | chair | 最大的椅子 | 0.1092 | 0.6982 | wrong_instance |
