# Sources and licensing

The integration and adapted engine code are distributed under GPL-3.0.

| Component | Source | Version / scope | License |
|---|---|---|---|
| YOLOv5 | https://github.com/ultralytics/yolov5 | v7.0, commit `915bbf294bb74c859f0b41f1c23bc395014ea679`; training engine and dataset augmentation | GPL-3.0 at the pinned revision; see LICENSE |
| FGD | https://github.com/yzd-v/FGD/blob/master/mmdet/distillation/losses/fgd.py | Yang et al., CVPR 2022; attention, foreground/background and learned relation loss | Apache-2.0; reference and full license retained under tests/reference |
| Classical KD concept | https://arxiv.org/abs/1503.02531 | Hinton et al.; temperature-scaled soft targets | Paper cited for the concept; no paper text or code copied |
| HaGRID | https://github.com/hukenovs/hagrid | Optional HaGRID v2 images and landmark annotations conversion example | Dataset is downloaded separately; retain its original terms and attribution |

The FGD adapter removes the MMDetection/MMCV dependency, uses PyTorch initialization, converts normalized YOLO targets, and clips boundary masks safely. The four original FGD terms are retained. Interior-box loss and gradient parity is checked against the unmodified reference using the same weights. This is an adaptation to YOLOv5, not a claim to reproduce the original COCO training scores.

The retained FGD reference file SHA256 is `769e7577d1fb2aedb52cf328fa95121a00c70af7fca0b65c8cfe3ff6363a35f8`.

The response KD baseline is an independently implemented detector adaptation of temperature-scaled binary soft-target cross entropy. It distills objectness and class logits with foreground/background balancing; it does not distill unaligned raw box logits and is not presented as an official detector paper implementation.
