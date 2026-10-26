# HaGRID 专项示例

此示例是通用工具的一套实验配方。使用自己的标准YOLO数据集时不需要执行本页的数据转换。

## 数据转换

自行按数据集许可获取HaGRID v2图片压缩包及annotation，然后运行：

```bash
distill-tool prepare-data --images /path/hagridv2_512.zip --annotations /path/annotations.zip --output data/hagrid10
```

转换器选取固定注释顺序的前十类，均衡生成80,000训练、10,000验证、20,000测试样本。三个划分来自官方对应划分，检查受试者、图片ID及内容哈希重叠；重复内容会跳过并补齐。类别顺序和数据来源保存在生成的清单中。

## 完整对照实验

先按README安装依赖，并准备固定版本引擎：

```bash
distill-tool bootstrap --engine vendor/yolov5
```

修改 `recipes/hagrid_320x328.yaml` 中的数据、引擎、工作目录和GPU编号，再运行：

```bash
distill-tool plan recipes/hagrid_320x328.yaml
distill-tool doctor recipes/hagrid_320x328.yaml
distill-tool run recipes/hagrid_320x328.yaml
```

这套配方使用m教师、n学生，依次执行教师、学生基线、Response KD、FGD、独立测试、测速、导出和报告。外部输入为320×328 RGB，内部补边到320×352。模型选择依据验证集；测试仅在配方冻结后执行。

## 显存与吞吐

该示例的batch256、workers32来自RTX6000 Ada 48GB的实测，不应直接当成其他设备的默认参数。需要调优时运行：

```bash
python scripts/profile_training.py --recipe recipes/hagrid_320x328.yaml --teacher /path/teacher.pt --device 0 --output training-capacity.json
python scripts/profile_loader.py --recipe recipes/hagrid_320x328.yaml --output loader-capacity.json
```

容量测量包含教师和蒸馏学生的训练计算。请在空闲设备上测量，按实际显存峰值和吞吐选择参数。公平对照应让各学生使用相同batch、轮数和优化设置；训练配方改变后使用新工作目录。

蒸馏效果与工程链路分开评估：比较蒸馏学生和相同结构基线的测试精度、同条件延迟；单次种子的微小差异不足以证明稳定提升。
