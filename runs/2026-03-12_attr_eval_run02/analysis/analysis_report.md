# Attribute Eval Analysis

## Overall

- num_tasks: 1854
- num_hit: 647
- acc: 0.349
- mean_iou_all: 0.3152

## Image-level Stats

- num_images: 179
- mean_image_acc: 0.3586
- p25_image_acc: 0.1667
- p50_image_acc: 0.3333
- p75_image_acc: 0.5
- min_image_acc: 0.0
- max_image_acc: 1.0

## By Attribute

| attribute_type | num_tasks | num_hit | acc | mean_iou |
| --- | --- | --- | --- | --- |
| size | 618 | 248 | 0.4013 | 0.3646 |
| spatial | 1236 | 399 | 0.3228 | 0.2906 |

## By Category

| category_en | num_tasks | num_hit | acc | mean_iou |
| --- | --- | --- | --- | --- |
| person | 582 | 118 | 0.2027 | 0.1975 |
| chair | 240 | 89 | 0.3708 | 0.3401 |
| bottle | 180 | 86 | 0.4778 | 0.4134 |
| car | 174 | 56 | 0.3218 | 0.2959 |
| cup | 174 | 64 | 0.3678 | 0.3397 |
| backpack | 132 | 51 | 0.3864 | 0.2952 |
| bicycle | 132 | 49 | 0.3712 | 0.3036 |
| laptop | 120 | 62 | 0.5167 | 0.4604 |
| dog | 114 | 66 | 0.5789 | 0.5555 |
| microwave | 6 | 6 | 1.0 | 0.8836 |

## Miss Reason Distribution

| miss_reason | count | ratio_in_miss | ratio_in_all_tasks |
| --- | --- | --- | --- |
| wrong_instance | 493 | 0.4085 | 0.2659 |
| low_iou | 482 | 0.3993 | 0.26 |
| no_box | 232 | 0.1922 | 0.1251 |

## Top-20 Low-IoU Tasks

| task_id | image_id | file_name | attribute_type | category_en | prompt_zh | best_iou | best_pred_score | miss_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 170670_107784_size_smallest | 170670 | 000000170670.jpg | size | chair | 最小的椅子 | 0.3998 | 0.3157 | low_iou |
| 170670_107784_spatial_rightmost | 170670 | 000000170670.jpg | spatial | chair | 最右侧的椅子 | 0.3983 | 0.3136 | low_iou |
| 13923_108331_spatial_leftmost | 13923 | 000000013923.jpg | spatial | chair | 最左侧的椅子 | 0.3754 | 0.3299 | low_iou |
| 568981_499140_spatial_bottommost | 568981 | 000000568981.jpg | spatial | person | 最下方的人 | 0.3753 | 0.361 | low_iou |
| 189213_1590733_spatial_topmost | 189213 | 000000189213.jpg | spatial | chair | 最上方的椅子 | 0.3707 | 0.4129 | low_iou |
| 189213_1590733_size_smallest | 189213 | 000000189213.jpg | size | chair | 最小的椅子 | 0.3701 | 0.4997 | low_iou |
| 149568_53166_spatial_leftmost | 149568 | 000000149568.jpg | spatial | dog | 最左侧的狗 | 0.3631 | 0.814 | low_iou |
| 189213_1590733_spatial_leftmost | 189213 | 000000189213.jpg | spatial | chair | 最左侧的椅子 | 0.3611 | 0.3584 | low_iou |
| 107226_18116_spatial_bottommost | 107226 | 000000107226.jpg | spatial | dog | 最下方的狗 | 0.3548 | 0.301 | low_iou |
| 107226_18116_spatial_leftmost | 107226 | 000000107226.jpg | spatial | dog | 最左侧的狗 | 0.3531 | 0.3207 | low_iou |
| 107226_18116_size_smallest | 107226 | 000000107226.jpg | size | dog | 最小的狗 | 0.3529 | 0.3205 | low_iou |
| 149568_53166_spatial_topmost | 149568 | 000000149568.jpg | spatial | dog | 最上方的狗 | 0.3518 | 0.7676 | low_iou |
| 324258_443107_spatial_rightmost | 324258 | 000000324258.jpg | spatial | person | 最右侧的人 | 0.3501 | 0.6132 | low_iou |
| 78748_1423049_size_smallest | 78748 | 000000078748.jpg | size | backpack | 最小的背包 | 0.3479 | 0.3057 | low_iou |
| 149568_53166_size_smallest | 149568 | 000000149568.jpg | size | dog | 最小的狗 | 0.3453 | 0.8488 | low_iou |
| 433103_1233577_size_largest | 433103 | 000000433103.jpg | size | person | 最大的人 | 0.3453 | 0.4694 | low_iou |
| 325483_86880_size_smallest | 325483 | 000000325483.jpg | size | bottle | 最小的瓶子 | 0.3418 | 0.3173 | low_iou |
| 463618_2165516_spatial_leftmost | 463618 | 000000463618.jpg | spatial | person | 最左侧的人 | 0.3336 | 0.4746 | low_iou |
| 91500_512958_size_smallest | 91500 | 000000091500.jpg | size | person | 最小的人 | 0.3286 | 0.3035 | low_iou |
| 279730_2155609_spatial_bottommost | 279730 | 000000279730.jpg | spatial | person | 最下方的人 | 0.3237 | 0.3106 | low_iou |

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
