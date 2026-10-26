# 通用训练工作台

## 安装与启动

前提：Python 3.10 或 3.11、Git；GPU 训练需有兼容 CUDA 12.1 的 NVIDIA 驱动。首次安装会下载依赖，需要网络。Linux GPU 是当前已经验收的训练环境；Windows 启动脚本和 UI 可用，完整 GPU 训练仍需在目标 Windows 机器验收。

Windows：双击仓库根目录 `start.bat`。首次运行会调用 `scripts/setup.ps1` 创建 `.venv`、安装依赖；以后直接启动界面。自定义 Python 路径：

```powershell
powershell -File scripts/setup.ps1 -Python C:\Python310\python.exe
```

Linux：

```bash
bash start.sh
```

首次安装可用 `PYTHON=/path/to/python3.10 bash scripts/setup.sh` 指定 Python；仅 CPU 安装运行 `bash scripts/setup.sh --cpu`。已经装好项目时可直接运行 `distill-tool ui`，浏览器默认打开 `http://127.0.0.1:8765`。

## 使用

1. 选择标准 YOLO 数据集的 `dataset.yaml`。最少需要 train、val、names；标签为 YOLO 归一化检测框，图片与标签置于对应 images/labels 目录。
2. 选择 YOLOv5 n/s/m/l/x 学生模型，选择常规训练、Response KD、FGD 或已安装插件。蒸馏时再选择教师模型。
3. 教师可先训练，也可直接提供已训练权重；已有教师的类别数量与顺序必须和数据集一致。空初始化权重表示使用对应官方预训练权重。
4. 设置 RGB 高宽、轮数、batch、单张 GPU、输出目录。默认 batch16 不代表所有显卡都适用；显存不足时降低 batch 或尺寸。
5. 按需勾选学生基线、独立测试、GPU 测速、ONNX 导出，点击“开始训练”。首次任务自动下载固定版本 YOLOv5 引擎，缺失预训练权重由引擎下载。

运行页显示阶段、当前轮数、实时日志和输出目录。任务记录写入 `~/.yolo-distill/jobs`，运行文件写入选择的输出目录。关闭浏览器后，已提交任务继续运行；重开工作台可以查看。停止会终止该任务进程树，恢复使用最近一轮的完整 checkpoint；首轮尚未保存 checkpoint 时需新建任务。

## 在 GPU 服务器运行

在服务器安装和启动服务：

```bash
distill-tool ui --no-browser --port 8765
```

本机通过 SSH 转发，再打开本机浏览器：

```bash
ssh -L 8765:127.0.0.1:8765 user@your-server
```

工作台浏览的文件和选择的 GPU 均位于启动服务的机器。服务仅监听回环地址，API 使用会话令牌并检查同源请求。任务进程独立于浏览器；电脑休眠会断开浏览器连接，服务器训练继续运行。

可选环境变量 `YOLO_DISTILL_ENGINE`、`YOLO_DISTILL_OUTPUT`、`YOLO_DISTILL_DEVICE` 分别设置当前实例的默认引擎路径、输出根目录和设备编号，不需要修改代码。

## 扩展方法

通过 `distill_tool.methods` entry point 安装的插件会自动出现在方法下拉框。内置方法提供参数表单；其他插件支持默认参数和 JSON 附加参数。插件必须使用 [方法接口](METHODS.md)，模型家族目前只支持固定版本 YOLOv5 n/s/m/l/x，其他检测器需要增加引擎适配。

## 发布范围

当前采用本地 Web 界面调用 Python 应用服务，共用同一训练后端；不依赖 Qt。GitHub 提供源码、界面、安装入口、文档与示例，数据和训练权重由用户自行准备。自动检查见 `tests/` 和 GitHub Actions。
