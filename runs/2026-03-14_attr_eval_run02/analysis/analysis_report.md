# Attribute Eval Analysis

## Overall

- num_tasks: 1854
- num_hit: 1022
- acc: 0.5512
- mean_iou_all: 0.4792

## Image-level Stats

- num_images: 179
- mean_image_acc: 0.5582
- p25_image_acc: 0.3333
- p50_image_acc: 0.5
- p75_image_acc: 0.75
- min_image_acc: 0.0
- max_image_acc: 1.0

## By Attribute

| attribute_type | num_tasks | num_hit | acc | mean_iou |
| --- | --- | --- | --- | --- |
| size | 618 | 312 | 0.5049 | 0.4356 |
| spatial | 1236 | 710 | 0.5744 | 0.501 |

## By Category

| category_en | num_tasks | num_hit | acc | mean_iou |
| --- | --- | --- | --- | --- |
| person | 582 | 313 | 0.5378 | 0.4752 |
| chair | 240 | 115 | 0.4792 | 0.4163 |
| bottle | 180 | 108 | 0.6 | 0.5308 |
| car | 174 | 68 | 0.3908 | 0.3528 |
| cup | 174 | 98 | 0.5632 | 0.4994 |
| backpack | 132 | 58 | 0.4394 | 0.3277 |
| bicycle | 132 | 80 | 0.6061 | 0.478 |
| laptop | 120 | 84 | 0.7 | 0.6277 |
| dog | 114 | 92 | 0.807 | 0.7111 |
| microwave | 6 | 6 | 1.0 | 0.9103 |

## Miss Reason Distribution

| miss_reason | count | ratio_in_miss | ratio_in_all_tasks |
| --- | --- | --- | --- |
| low_iou | 443 | 0.5325 | 0.2389 |
| wrong_instance | 329 | 0.3954 | 0.1775 |
| no_box | 60 | 0.0721 | 0.0324 |

## Top-20 Low-IoU Tasks

| task_id | image_id | file_name | attribute_type | category_en | prompt_zh | best_iou | best_pred_score | miss_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 465822_1727101_spatial_leftmost | 465822 | 000000465822.jpg | spatial | person | 最左侧的人 | 0.2994 | 0.5839 | low_iou |
| 38829_128391_size_smallest | 38829 | 000000038829.jpg | size | bicycle | 最小的自行车 | 0.2937 | 0.8602 | low_iou |
| 38829_128391_spatial_rightmost | 38829 | 000000038829.jpg | spatial | bicycle | 最右侧的自行车 | 0.2937 | 0.8602 | low_iou |
| 38829_128391_spatial_bottommost | 38829 | 000000038829.jpg | spatial | bicycle | 最下方的自行车 | 0.2937 | 0.8602 | low_iou |
| 483999_546252_size_largest | 483999 | 000000483999.jpg | size | person | 最大的人 | 0.2934 | 0.6292 | low_iou |
| 198641_2134578_size_smallest | 198641 | 000000198641.jpg | size | laptop | 最小的笔记本电脑 | 0.2863 | 0.8446 | low_iou |
| 198641_2134578_spatial_rightmost | 198641 | 000000198641.jpg | spatial | laptop | 最右侧的笔记本电脑 | 0.2863 | 0.8446 | low_iou |
| 198641_2134578_spatial_bottommost | 198641 | 000000198641.jpg | spatial | laptop | 最下方的笔记本电脑 | 0.2863 | 0.8446 | low_iou |
| 508370_1204446_spatial_topmost | 508370 | 000000508370.jpg | spatial | person | 最上方的人 | 0.2846 | 0.6902 | low_iou |
| 463522_1235913_spatial_rightmost | 463522 | 000000463522.jpg | spatial | person | 最右侧的人 | 0.2731 | 0.5802 | low_iou |
| 433103_1296539_spatial_bottommost | 433103 | 000000433103.jpg | spatial | person | 最下方的人 | 0.2703 | 0.7087 | low_iou |
| 346232_469928_size_largest | 346232 | 000000346232.jpg | size | person | 最大的人 | 0.2691 | 0.6241 | low_iou |
| 245173_1335240_size_smallest | 245173 | 000000245173.jpg | size | bicycle | 最小的自行车 | 0.2683 | 0.6171 | low_iou |
| 245173_1335240_spatial_leftmost | 245173 | 000000245173.jpg | spatial | bicycle | 最左侧的自行车 | 0.2683 | 0.6171 | low_iou |
| 245173_1335240_spatial_topmost | 245173 | 000000245173.jpg | spatial | bicycle | 最上方的自行车 | 0.2683 | 0.6171 | low_iou |
| 357978_1588937_spatial_leftmost | 357978 | 000000357978.jpg | spatial | chair | 最左侧的椅子 | 0.2612 | 0.5979 | low_iou |
| 107226_216223_spatial_leftmost | 107226 | 000000107226.jpg | spatial | person | 最左侧的人 | 0.2587 | 0.6215 | low_iou |
| 100428_510346_spatial_leftmost | 100428 | 000000100428.jpg | spatial | person | 最左侧的人 | 0.2535 | 0.6346 | low_iou |
| 100428_510346_spatial_topmost | 100428 | 000000100428.jpg | spatial | person | 最上方的人 | 0.2535 | 0.6346 | low_iou |
| 57150_2159736_size_largest | 57150 | 000000057150.jpg | size | person | 最大的人 | 0.2529 | 0.6333 | low_iou |

## Top-20 Wrong-Instance Tasks

| task_id | image_id | file_name | attribute_type | category_en | prompt_zh | best_iou | best_pred_score | miss_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 97337_2032606_size_smallest | 97337 | 000000097337.jpg | size | person | 最小的人 | 0.1938 | 0.7063 | wrong_instance |
| 97337_2032606_spatial_leftmost | 97337 | 000000097337.jpg | spatial | person | 最左侧的人 | 0.1938 | 0.7063 | wrong_instance |
| 97337_2032606_spatial_bottommost | 97337 | 000000097337.jpg | spatial | person | 最下方的人 | 0.1938 | 0.7063 | wrong_instance |
| 84650_1823740_size_smallest | 84650 | 000000084650.jpg | size | backpack | 最小的背包 | 0.1876 | 0.6887 | wrong_instance |
| 508370_501765_spatial_bottommost | 508370 | 000000508370.jpg | spatial | person | 最下方的人 | 0.1856 | 0.6013 | wrong_instance |
| 380913_524668_spatial_leftmost | 380913 | 000000380913.jpg | spatial | person | 最左侧的人 | 0.1839 | 0.6938 | wrong_instance |
| 11197_2157164_spatial_bottommost | 11197 | 000000011197.jpg | spatial | person | 最下方的人 | 0.1829 | 0.7045 | wrong_instance |
| 269113_18383_spatial_rightmost | 269113 | 000000269113.jpg | spatial | dog | 最右侧的狗 | 0.1817 | 0.8484 | wrong_instance |
| 493286_2027667_size_smallest | 493286 | 000000493286.jpg | size | person | 最小的人 | 0.1598 | 0.5898 | wrong_instance |
| 493286_2027667_spatial_topmost | 493286 | 000000493286.jpg | spatial | person | 最上方的人 | 0.1598 | 0.5898 | wrong_instance |
| 100723_1764004_spatial_leftmost | 100723 | 000000100723.jpg | spatial | person | 最左侧的人 | 0.1397 | 0.6211 | wrong_instance |
| 74058_344286_spatial_topmost | 74058 | 000000074058.jpg | spatial | bicycle | 最上方的自行车 | 0.1344 | 0.6296 | wrong_instance |
| 397354_1710302_size_smallest | 397354 | 000000397354.jpg | size | person | 最小的人 | 0.134 | 0.6106 | wrong_instance |
| 397354_1710302_spatial_rightmost | 397354 | 000000397354.jpg | spatial | person | 最右侧的人 | 0.134 | 0.6106 | wrong_instance |
| 89880_6105_size_largest | 89880 | 000000089880.jpg | size | dog | 最大的狗 | 0.1277 | 0.8077 | wrong_instance |
| 89880_6127_size_smallest | 89880 | 000000089880.jpg | size | dog | 最小的狗 | 0.1265 | 0.7703 | wrong_instance |
| 411938_193398_size_largest | 411938 | 000000411938.jpg | size | person | 最大的人 | 0.1202 | 0.6489 | wrong_instance |
| 74058_344274_spatial_bottommost | 74058 | 000000074058.jpg | spatial | bicycle | 最下方的自行车 | 0.1161 | 0.6195 | wrong_instance |
| 325838_2195234_size_smallest | 325838 | 000000325838.jpg | size | laptop | 最小的笔记本电脑 | 0.1159 | 0.6264 | wrong_instance |
| 520659_1592232_spatial_topmost | 520659 | 000000520659.jpg | spatial | chair | 最上方的椅子 | 0.1139 | 0.5817 | wrong_instance |
