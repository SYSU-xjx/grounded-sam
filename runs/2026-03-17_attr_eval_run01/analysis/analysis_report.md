# Attribute Eval Analysis

## Overall

- num_tasks: 1854
- num_hit: 773
- acc: 0.4169
- mean_iou_all: 0.363

## Image-level Stats

- num_images: 179
- mean_image_acc: 0.4214
- p25_image_acc: 0.25
- p50_image_acc: 0.4167
- p75_image_acc: 0.5
- min_image_acc: 0.0
- max_image_acc: 1.0

## By Attribute

| attribute_type | num_tasks | num_hit | acc | mean_iou |
| --- | --- | --- | --- | --- |
| size | 618 | 271 | 0.4385 | 0.3809 |
| spatial | 1236 | 502 | 0.4061 | 0.354 |

## By Category

| category_en | num_tasks | num_hit | acc | mean_iou |
| --- | --- | --- | --- | --- |
| person | 582 | 231 | 0.3969 | 0.36 |
| chair | 240 | 89 | 0.3708 | 0.3214 |
| bottle | 180 | 77 | 0.4278 | 0.3724 |
| car | 174 | 56 | 0.3218 | 0.289 |
| cup | 174 | 70 | 0.4023 | 0.359 |
| backpack | 132 | 55 | 0.4167 | 0.3012 |
| bicycle | 132 | 55 | 0.4167 | 0.3344 |
| laptop | 120 | 59 | 0.4917 | 0.4389 |
| dog | 114 | 75 | 0.6579 | 0.5667 |
| microwave | 6 | 6 | 1.0 | 0.8826 |

## Miss Reason Distribution

| miss_reason | count | ratio_in_miss | ratio_in_all_tasks |
| --- | --- | --- | --- |
| wrong_instance | 642 | 0.5939 | 0.3463 |
| low_iou | 394 | 0.3645 | 0.2125 |
| no_box | 45 | 0.0416 | 0.0243 |

## Top-20 Low-IoU Tasks

| task_id | image_id | file_name | attribute_type | category_en | prompt_zh | best_iou | best_pred_score | miss_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 138550_374914_spatial_topmost | 138550 | 000000138550.jpg | spatial | chair | topmost chair | 0.2976 | 0.6443 | low_iou |
| 138550_374914_spatial_rightmost | 138550 | 000000138550.jpg | spatial | chair | rightmost chair | 0.2971 | 0.6126 | low_iou |
| 397354_1710302_size_smallest | 397354 | 000000397354.jpg | size | person | smallest person | 0.2955 | 0.3735 | low_iou |
| 138550_374914_size_smallest | 138550 | 000000138550.jpg | size | chair | smallest chair | 0.2946 | 0.6116 | low_iou |
| 397354_1882278_size_smallest | 397354 | 000000397354.jpg | size | cup | smallest cup | 0.2942 | 0.3004 | low_iou |
| 38829_128391_spatial_rightmost | 38829 | 000000038829.jpg | spatial | bicycle | rightmost bicycle | 0.2923 | 0.8088 | low_iou |
| 38829_128391_spatial_bottommost | 38829 | 000000038829.jpg | spatial | bicycle | bottommost bicycle | 0.2909 | 0.8453 | low_iou |
| 14439_375587_spatial_bottommost | 14439 | 000000014439.jpg | spatial | chair | bottommost chair | 0.2881 | 0.3696 | low_iou |
| 14439_375587_spatial_rightmost | 14439 | 000000014439.jpg | spatial | chair | rightmost chair | 0.2875 | 0.3514 | low_iou |
| 38829_128391_size_smallest | 38829 | 000000038829.jpg | size | bicycle | smallest bicycle | 0.2868 | 0.9023 | low_iou |
| 410880_2223427_size_largest | 410880 | 000000410880.jpg | size | chair | largest chair | 0.286 | 0.315 | low_iou |
| 94944_1825525_spatial_rightmost | 94944 | 000000094944.jpg | spatial | backpack | rightmost backpack | 0.2854 | 0.3687 | low_iou |
| 94944_1825525_spatial_bottommost | 94944 | 000000094944.jpg | spatial | backpack | bottommost backpack | 0.2849 | 0.3661 | low_iou |
| 198641_2134578_size_smallest | 198641 | 000000198641.jpg | size | laptop | smallest laptop | 0.2832 | 0.8573 | low_iou |
| 94944_1825525_size_largest | 94944 | 000000094944.jpg | size | backpack | largest backpack | 0.2825 | 0.4132 | low_iou |
| 198641_2134578_spatial_rightmost | 198641 | 000000198641.jpg | spatial | laptop | rightmost laptop | 0.2821 | 0.6986 | low_iou |
| 198641_2134578_spatial_bottommost | 198641 | 000000198641.jpg | spatial | laptop | bottommost laptop | 0.2811 | 0.6711 | low_iou |
| 14439_375587_size_smallest | 14439 | 000000014439.jpg | size | chair | smallest chair | 0.2804 | 0.4036 | low_iou |
| 356248_542388_size_smallest | 356248 | 000000356248.jpg | size | person | smallest person | 0.2782 | 0.6255 | low_iou |
| 256941_130736_size_smallest | 256941 | 000000256941.jpg | size | bicycle | smallest bicycle | 0.2691 | 0.344 | low_iou |

## Top-20 Wrong-Instance Tasks

| task_id | image_id | file_name | attribute_type | category_en | prompt_zh | best_iou | best_pred_score | miss_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 183716_455359_size_largest | 183716 | 000000183716.jpg | size | person | largest person | 0.1975 | 0.6392 | wrong_instance |
| 84650_1823740_spatial_leftmost | 84650 | 000000084650.jpg | spatial | backpack | leftmost backpack | 0.1848 | 0.3282 | wrong_instance |
| 84650_1823740_spatial_topmost | 84650 | 000000084650.jpg | spatial | backpack | topmost backpack | 0.1845 | 0.3345 | wrong_instance |
| 84650_1823740_size_smallest | 84650 | 000000084650.jpg | size | backpack | smallest backpack | 0.1842 | 0.3063 | wrong_instance |
| 269113_18383_size_smallest | 269113 | 000000269113.jpg | size | dog | smallest dog | 0.1781 | 0.3481 | wrong_instance |
| 269113_18383_spatial_rightmost | 269113 | 000000269113.jpg | spatial | dog | rightmost dog | 0.1775 | 0.3936 | wrong_instance |
| 74058_344274_spatial_leftmost | 74058 | 000000074058.jpg | spatial | bicycle | leftmost bicycle | 0.1694 | 0.4659 | wrong_instance |
| 397354_254502_spatial_bottommost | 397354 | 000000397354.jpg | spatial | person | bottommost person | 0.1692 | 0.4118 | wrong_instance |
| 74058_344274_spatial_bottommost | 74058 | 000000074058.jpg | spatial | bicycle | bottommost bicycle | 0.1684 | 0.4853 | wrong_instance |
| 74058_344274_size_smallest | 74058 | 000000074058.jpg | size | bicycle | smallest bicycle | 0.1648 | 0.6829 | wrong_instance |
| 13923_1490683_spatial_bottommost | 13923 | 000000013923.jpg | spatial | bottle | bottommost bottle | 0.1645 | 0.3237 | wrong_instance |
| 13923_1490683_spatial_leftmost | 13923 | 000000013923.jpg | spatial | bottle | leftmost bottle | 0.1637 | 0.3259 | wrong_instance |
| 11197_2157164_spatial_bottommost | 11197 | 000000011197.jpg | spatial | person | bottommost person | 0.162 | 0.583 | wrong_instance |
| 493286_2027667_spatial_topmost | 493286 | 000000493286.jpg | spatial | person | topmost person | 0.1568 | 0.3369 | wrong_instance |
| 84241_490280_size_smallest | 84241 | 000000084241.jpg | size | person | smallest person | 0.1564 | 0.4303 | wrong_instance |
| 397354_260673_size_largest | 397354 | 000000397354.jpg | size | person | largest person | 0.1508 | 0.3024 | wrong_instance |
| 286553_445938_spatial_topmost | 286553 | 000000286553.jpg | spatial | person | topmost person | 0.1469 | 0.4668 | wrong_instance |
| 286553_445938_spatial_leftmost | 286553 | 000000286553.jpg | spatial | person | leftmost person | 0.1467 | 0.4811 | wrong_instance |
| 286553_445938_size_smallest | 286553 | 000000286553.jpg | size | person | smallest person | 0.1465 | 0.6606 | wrong_instance |
| 84241_490280_spatial_topmost | 84241 | 000000084241.jpg | spatial | person | topmost person | 0.1405 | 0.3851 | wrong_instance |
