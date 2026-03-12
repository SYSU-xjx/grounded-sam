# Attribute Eval Analysis

## Overall

- num_tasks: 1854
- num_hit: 671
- acc: 0.3619
- mean_iou_all: 0.3152

## Image-level Stats

- num_images: 179
- mean_image_acc: 0.3726
- p25_image_acc: 0.1667
- p50_image_acc: 0.3333
- p75_image_acc: 0.5
- min_image_acc: 0.0
- max_image_acc: 1.0

## By Attribute

| attribute_type | num_tasks | num_hit | acc | mean_iou |
| --- | --- | --- | --- | --- |
| size | 618 | 258 | 0.4175 | 0.3646 |
| spatial | 1236 | 413 | 0.3341 | 0.2906 |

## By Category

| category_en | num_tasks | num_hit | acc | mean_iou |
| --- | --- | --- | --- | --- |
| person | 582 | 124 | 0.2131 | 0.1975 |
| chair | 240 | 97 | 0.4042 | 0.3401 |
| bottle | 180 | 87 | 0.4833 | 0.4134 |
| car | 174 | 56 | 0.3218 | 0.2959 |
| cup | 174 | 64 | 0.3678 | 0.3397 |
| backpack | 132 | 54 | 0.4091 | 0.2952 |
| bicycle | 132 | 49 | 0.3712 | 0.3036 |
| laptop | 120 | 62 | 0.5167 | 0.4604 |
| dog | 114 | 72 | 0.6316 | 0.5555 |
| microwave | 6 | 6 | 1.0 | 0.8836 |

## Miss Reason Distribution

| miss_reason | count | ratio_in_miss | ratio_in_all_tasks |
| --- | --- | --- | --- |
| wrong_instance | 493 | 0.4167 | 0.2659 |
| low_iou | 458 | 0.3872 | 0.247 |
| no_box | 232 | 0.1961 | 0.1251 |

## Top-20 Low-IoU Tasks

| task_id | image_id | file_name | attribute_type | category_en | prompt_zh | best_iou | best_pred_score | miss_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 138550_374914_spatial_rightmost | 138550 | 000000138550.jpg | spatial | chair | 最右侧的椅子 | 0.2992 | 0.5969 | low_iou |
| 138550_374914_spatial_topmost | 138550 | 000000138550.jpg | spatial | chair | 最上方的椅子 | 0.2988 | 0.5854 | low_iou |
| 138550_374914_size_smallest | 138550 | 000000138550.jpg | size | chair | 最小的椅子 | 0.2946 | 0.6116 | low_iou |
| 397354_1882278_size_smallest | 397354 | 000000397354.jpg | size | cup | 最小的杯子 | 0.2942 | 0.3004 | low_iou |
| 14439_375587_spatial_bottommost | 14439 | 000000014439.jpg | spatial | chair | 最下方的椅子 | 0.2911 | 0.4404 | low_iou |
| 38829_128391_spatial_bottommost | 38829 | 000000038829.jpg | spatial | bicycle | 最下方的自行车 | 0.2903 | 0.6896 | low_iou |
| 38829_128391_spatial_rightmost | 38829 | 000000038829.jpg | spatial | bicycle | 最右侧的自行车 | 0.2892 | 0.7817 | low_iou |
| 356248_542388_size_smallest | 356248 | 000000356248.jpg | size | person | 最小的人 | 0.2881 | 0.3437 | low_iou |
| 448076_198115_spatial_bottommost | 448076 | 000000448076.jpg | spatial | person | 最下方的人 | 0.288 | 0.3553 | low_iou |
| 38829_128391_size_smallest | 38829 | 000000038829.jpg | size | bicycle | 最小的自行车 | 0.2867 | 0.803 | low_iou |
| 198641_2134578_size_smallest | 198641 | 000000198641.jpg | size | laptop | 最小的笔记本电脑 | 0.2832 | 0.8573 | low_iou |
| 198641_2134578_spatial_rightmost | 198641 | 000000198641.jpg | spatial | laptop | 最右侧的笔记本电脑 | 0.2831 | 0.8985 | low_iou |
| 94944_1825525_spatial_rightmost | 94944 | 000000094944.jpg | spatial | backpack | 最右侧的背包 | 0.2822 | 0.3602 | low_iou |
| 94944_1825525_size_largest | 94944 | 000000094944.jpg | size | backpack | 最大的背包 | 0.2822 | 0.4079 | low_iou |
| 94944_1825525_spatial_bottommost | 94944 | 000000094944.jpg | spatial | backpack | 最下方的背包 | 0.2815 | 0.3567 | low_iou |
| 14439_375587_size_smallest | 14439 | 000000014439.jpg | size | chair | 最小的椅子 | 0.2804 | 0.4036 | low_iou |
| 198641_2134578_spatial_bottommost | 198641 | 000000198641.jpg | spatial | laptop | 最下方的笔记本电脑 | 0.2789 | 0.818 | low_iou |
| 14439_375587_spatial_rightmost | 14439 | 000000014439.jpg | spatial | chair | 最右侧的椅子 | 0.2787 | 0.4067 | low_iou |
| 245173_1335240_spatial_topmost | 245173 | 000000245173.jpg | spatial | bicycle | 最上方的自行车 | 0.2722 | 0.3069 | low_iou |
| 256941_130736_size_smallest | 256941 | 000000256941.jpg | size | bicycle | 最小的自行车 | 0.268 | 0.345 | low_iou |

## Top-20 Wrong-Instance Tasks

| task_id | image_id | file_name | attribute_type | category_en | prompt_zh | best_iou | best_pred_score | miss_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 183716_455359_size_largest | 183716 | 000000183716.jpg | size | person | 最大的人 | 0.1973 | 0.7258 | wrong_instance |
| 520659_1931072_spatial_leftmost | 520659 | 000000520659.jpg | spatial | chair | 最左侧的椅子 | 0.197 | 0.4186 | wrong_instance |
| 242060_1501568_spatial_topmost | 242060 | 000000242060.jpg | spatial | cup | 最上方的杯子 | 0.1859 | 0.3048 | wrong_instance |
| 84650_1823740_size_smallest | 84650 | 000000084650.jpg | size | backpack | 最小的背包 | 0.1842 | 0.3063 | wrong_instance |
| 269113_18383_spatial_rightmost | 269113 | 000000269113.jpg | spatial | dog | 最右侧的狗 | 0.1783 | 0.3573 | wrong_instance |
| 269113_18383_size_smallest | 269113 | 000000269113.jpg | size | dog | 最小的狗 | 0.1781 | 0.3481 | wrong_instance |
| 74058_344274_spatial_bottommost | 74058 | 000000074058.jpg | spatial | bicycle | 最下方的自行车 | 0.1665 | 0.394 | wrong_instance |
| 74058_344274_spatial_leftmost | 74058 | 000000074058.jpg | spatial | bicycle | 最左侧的自行车 | 0.1646 | 0.5454 | wrong_instance |
| 74058_344274_size_smallest | 74058 | 000000074058.jpg | size | bicycle | 最小的自行车 | 0.1626 | 0.6133 | wrong_instance |
| 286553_445938_spatial_leftmost | 286553 | 000000286553.jpg | spatial | person | 最左侧的人 | 0.1472 | 0.4245 | wrong_instance |
| 286553_84515_spatial_leftmost | 286553 | 000000286553.jpg | spatial | bottle | 最左侧的瓶子 | 0.1403 | 0.659 | wrong_instance |
| 286553_84515_spatial_bottommost | 286553 | 000000286553.jpg | spatial | bottle | 最下方的瓶子 | 0.14 | 0.7023 | wrong_instance |
| 286553_84515_size_smallest | 286553 | 000000286553.jpg | size | bottle | 最小的瓶子 | 0.1384 | 0.7402 | wrong_instance |
| 486040_1103890_size_smallest | 486040 | 000000486040.jpg | size | laptop | 最小的笔记本电脑 | 0.1382 | 0.5698 | wrong_instance |
| 486040_1103890_spatial_topmost | 486040 | 000000486040.jpg | spatial | laptop | 最上方的笔记本电脑 | 0.1381 | 0.5736 | wrong_instance |
| 486040_1103890_spatial_leftmost | 486040 | 000000486040.jpg | spatial | laptop | 最左侧的笔记本电脑 | 0.1376 | 0.522 | wrong_instance |
| 397354_260673_size_largest | 397354 | 000000397354.jpg | size | person | 最大的人 | 0.1332 | 0.7723 | wrong_instance |
| 411938_193398_size_largest | 411938 | 000000411938.jpg | size | person | 最大的人 | 0.1239 | 0.4033 | wrong_instance |
| 32610_1103853_spatial_rightmost | 32610 | 000000032610.jpg | spatial | laptop | 最右侧的笔记本电脑 | 0.1162 | 0.3055 | wrong_instance |
| 256916_382283_size_largest | 256916 | 000000256916.jpg | size | chair | 最大的椅子 | 0.1092 | 0.497 | wrong_instance |
