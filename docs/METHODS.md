# 方法接口与实现范围

`DistillationMethod` 是 `torch.nn.Module`，输入 `KDInputs`，输出 `(标量损失, 分项字典)`。损失为每图均值；训练器乘实际 batch_size 后与原 YOLO 检测损失相加。每方法的外部权重由 YAML 指定。

`KDInputs` 包含学生/教师各尺度原始输出、P3/P4/P5 检测头输入特征、YOLO格式归一化目标和**补边后**输入高宽。教师前向使用 `no_grad`，学生特征保留计算图；FGD内部的教师侧关系变换是可训练损失模块，与冻结的教师检测器区分。

## response_kd

参考 [Hinton et al.](https://arxiv.org/abs/1503.02531) 的温度软目标原则。YOLO使用独立sigmoid输出，因此采用逐logit binary cross entropy，双方logit除以温度，损失乘温度平方。objectness按教师置信度划分前景/背景并分别求均值，class只在前景蒸馏。

这是清楚命名的检测适配基线，带前背景平衡，不冒充某个检测论文的原实现。框回归保留原监督CIoU损失，不模仿语义不一致的原始框logit。启动时核对师生类别顺序、stride、anchor几何和输出尺寸。

## fgd

参考 [CVPR 2022论文](https://openaccess.thecvf.com/content/CVPR2022/html/Yang_Focal_and_Global_Knowledge_Distillation_for_Detectors_CVPR_2022_paper.html)和[作者代码](https://github.com/yzd-v/FGD)。保留四个损失：

1. 前景与背景的空间/通道注意力加权特征差异；前景按框在特征图的面积倒数加权，背景归一化。
2. 学生与教师的空间、通道注意力L1差异。
3. 全局关系分支：可训练空间context池化及channel-add变换后的特征差异。
4. 每层1×1对齐，P3/P4/P5损失相加，默认系数与作者实现一致。

YOLO适配：归一化目标转换到特征网格；边界安全裁剪；MMCV初始化替换为等价PyTorch初始化。float32计算FGD避免AMP大求和溢出。梯度及损失使用相同参数与作者参考实现对齐；不宣称复现原COCO训练指标。

## 添加一个外部插件

```python
import torch
from distill_tool.methods.base import DistillationMethod

class MyMethod(DistillationMethod):
    requires_features = True

    def __init__(self, student_channels, teacher_channels, scale=1.0):
        super().__init__()
        # 在这里注册通道对齐等训练参数，禁止将教师挂到学生模型内。
        self.scale = scale

    def forward(self, inputs):
        # 返回损失并保留学生梯度；teacher tensors按需detach。
        raise NotImplementedError
```

外部包的 `pyproject.toml`：

```toml
[project.entry-points."distill_tool.methods"]
my_method = "my_package:MyMethod"
```

安装后，工作台的方法下拉框自动出现 `my_method`；也可在recipe的 `methods` 下加入该方法及参数。插件标识使用小写字母、数字和下划线。当前YOLO适配器可提供上述输出与特征；超出这些输入需求时扩展引擎接口并增加对应测试，不绕过能力检查。

每个插件的验收必须包括数值/梯度、空目标/边界、教师冻结、方法参数被优化和保存/恢复、真实短训练、纯学生部署。算法精度效果独立报告。
