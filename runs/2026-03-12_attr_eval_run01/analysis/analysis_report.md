# Attribute Eval Analysis

## Overall

- num_tasks: 1854
- num_hit: 616
- acc: 0.3323
- mean_iou_all: 0.3152

## Image-level Stats

- num_images: 179
- mean_image_acc: 0.3405
- p25_image_acc: 0.1667
- p50_image_acc: 0.3333
- p75_image_acc: 0.5
- min_image_acc: 0.0
- max_image_acc: 1.0

## By Attribute

| attribute_type | num_tasks | num_hit | acc | mean_iou |
| --- | --- | --- | --- | --- |
| size | 618 | 236 | 0.3819 | 0.3646 |
| spatial | 1236 | 380 | 0.3074 | 0.2906 |

## By Category

| category_en | num_tasks | num_hit | acc | mean_iou |
| --- | --- | --- | --- | --- |
| person | 582 | 118 | 0.2027 | 0.1975 |
| chair | 240 | 85 | 0.3542 | 0.3401 |
| bottle | 180 | 82 | 0.4556 | 0.4134 |
| car | 174 | 56 | 0.3218 | 0.2959 |
| cup | 174 | 64 | 0.3678 | 0.3397 |
| backpack | 132 | 42 | 0.3182 | 0.2952 |
| bicycle | 132 | 40 | 0.303 | 0.3036 |
| laptop | 120 | 60 | 0.5 | 0.4604 |
| dog | 114 | 63 | 0.5526 | 0.5555 |
| microwave | 6 | 6 | 1.0 | 0.8836 |

## Miss Reason Distribution

| miss_reason | count | ratio_in_miss | ratio_in_all_tasks |
| --- | --- | --- | --- |
| low_iou | 513 | 0.4144 | 0.2767 |
| wrong_instance | 493 | 0.3982 | 0.2659 |
| no_box | 232 | 0.1874 | 0.1251 |

## Top-20 Failed Tasks (lowest IoU)

| task_id | image_id | file_name | attribute_type | category_en | prompt_zh | best_iou | best_pred_score | miss_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 9400_1294548_size_smallest | 9400 | 000000009400.jpg | size | person | 最小的人 | 0.0 | 0.0 | no_box |
| 9400_1294548_spatial_topmost | 9400 | 000000009400.jpg | spatial | person | 最上方的人 | 0.0 | 0.0 | no_box |
| 9400_1292246_spatial_bottommost | 9400 | 000000009400.jpg | spatial | person | 最下方的人 | 0.0 | 0.0 | no_box |
| 11197_1237643_size_smallest | 11197 | 000000011197.jpg | size | person | 最小的人 | 0.0 | 0.0 | no_box |
| 11197_1237643_spatial_topmost | 11197 | 000000011197.jpg | spatial | person | 最上方的人 | 0.0 | 0.0 | no_box |
| 11197_2157164_spatial_bottommost | 11197 | 000000011197.jpg | spatial | person | 最下方的人 | 0.0 | 0.0 | no_box |
| 13659_2008612_size_smallest | 13659 | 000000013659.jpg | size | person | 最小的人 | 0.0 | 0.0 | no_box |
| 13659_201643_spatial_topmost | 13659 | 000000013659.jpg | spatial | person | 最上方的人 | 0.0 | 0.0 | no_box |
| 13659_199407_spatial_bottommost | 13659 | 000000013659.jpg | spatial | person | 最下方的人 | 0.0 | 0.0 | no_box |
| 14439_1170759_size_smallest | 14439 | 000000014439.jpg | size | backpack | 最小的背包 | 0.0 | 0.0 | no_box |
| 14439_1827291_spatial_leftmost | 14439 | 000000014439.jpg | spatial | backpack | 最左侧的背包 | 0.0 | 0.0 | no_box |
| 14439_1827291_spatial_topmost | 14439 | 000000014439.jpg | spatial | backpack | 最上方的背包 | 0.0 | 0.0 | no_box |
| 14439_1826731_spatial_bottommost | 14439 | 000000014439.jpg | spatial | backpack | 最下方的背包 | 0.0 | 0.0 | no_box |
| 14439_1755416_size_smallest | 14439 | 000000014439.jpg | size | person | 最小的人 | 0.0 | 0.0 | no_box |
| 14439_1733380_spatial_bottommost | 14439 | 000000014439.jpg | spatial | person | 最下方的人 | 0.0 | 0.0 | no_box |
| 25394_2206285_spatial_topmost | 25394 | 000000025394.jpg | spatial | person | 最上方的人 | 0.0 | 0.0 | no_box |
| 25394_1495015_size_smallest | 25394 | 000000025394.jpg | size | bottle | 最小的瓶子 | 0.0 | 0.0 | no_box |
| 25394_1494849_spatial_leftmost | 25394 | 000000025394.jpg | spatial | bottle | 最左侧的瓶子 | 0.0 | 0.0 | no_box |
| 25394_2095304_spatial_rightmost | 25394 | 000000025394.jpg | spatial | bottle | 最右侧的瓶子 | 0.0 | 0.0 | no_box |
| 29984_529789_size_smallest | 29984 | 000000029984.jpg | size | person | 最小的人 | 0.0 | 0.0 | no_box |
