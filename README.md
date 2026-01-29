# 🤖 X-VLA：基于软提示Transformer的可扩展跨具身视觉-语言-动作模型
| 📄 **论文** | 🌐 **项目主页** | 🤗 **Hugging Face 平台** |
| :---: | :---: | :---: |
| [阅读完整研究论文](https://arxiv.org/pdf/2510.10274) | [查看演示案例](https://thu-air-dream.github.io/X-VLA/) | [获取模型与数据集](https://huggingface.co/collections/2toINF/x-vla) |

## 🏆 亮点与动态

### 🎉 重磅喜讯：X-VLA 被 ICLR 2026 接收
我们非常荣幸地宣布，**X-VLA 成功入选 ICLR 2026 会议**。

### 🚀 现已接入 LeRobot 框架
X-VLA 已原生集成至 [LeRobot 平台](https://huggingface.co/docs/lerobot/xvla)。
欢迎大家试用！衷心感谢 Hugging Face 团队的支持与协作。

### 🥇 斩获 IROS 2025 赛事冠军
X-VLA 在 **2025 年国际智能机器人与系统大会（IROS 2025）** 举办的 **AgiBot 世界挑战赛** 中，荣获**第一名（冠军）**。

---

## 🧩 概述
成功的通用型**视觉-语言-动作（VLA）模型**，依赖于在多样化机器人具身形态下，开展可扩展的跨平台训练。
为充分利用大规模机器人数据集的异构特性，**X-VLA** 提出了**软提示**机制——通过针对不同具身形态设计的可学习嵌入向量，引导统一的Transformer主干网络，实现高效的多领域策略学习。

所构建的**X-VLA-0.9B**模型架构，在六大仿真平台与三台实体机器人上实现了**业界领先的泛化能力**，在操作灵巧性、环境适应性与运行效率上，均超越了过往的视觉-语言-动作模型方案。

---

## 🚀 快速上手：安装与部署

### 1️⃣ 环境安装
```bash
# 克隆代码仓库
git clone https://github.com/2toinf/X-VLA.git
cd X-VLA
```

```bash
# 创建并激活 Conda 虚拟环境
conda create -n XVLA python=3.10 -y
conda activate XVLA

# 安装依赖库
pip install -r requirements.txt
```

或执行以下命令：
```bash
conda env create -f environment.yml
conda activate xvla-stable
```

---
### 2️⃣ X-VLA 推理部署
X-VLA 采用**服务端-客户端**架构，将模型运行环境与仿真、机器人专属依赖库相互隔离。
该设计能够避免软件包冲突，同时支持在多GPU、SLURM集群或边缘设备上开展分布式推理。

#### 🧠 可用预训练模型
- [ ] 我们发现，将模型转换为Hugging Face格式后，在不同数据集上的性能出现小幅下降（降幅约1%），目前正在排查问题原因。

#### 🧠 Libero 环境配置与评估说明
- [x] 若你有关于相对动作转换为绝对动作、以及相关代码实现的疑问，请先查阅议题 [#2](https://github.com/2toinf/X-VLA/issues/2) 与 [#15](https://github.com/2toinf/X-VLA/issues/15)。我们已在[此处](https://github.com/2toinf/X-VLA/blob/main/evaluation/libero/preprocess.md)更新了完整的数据预处理指南。

#### 🔥 更新说明：我们已开源LoRA微调代码、模型权重，以及配套的推理代码。

| 模型ID | 具身形态 | 说明 | 性能指标 | 评估指南 |
| :------------------------------------------------------------------------------------------------- | :---------------- | :------------------------------------------------------------------------------ | :--------------: | :-----------------: |
| [`2toINF/X-VLA-Pt`](https://huggingface.co/2toINF/X-VLA-Pt) | 基础模型 | 在大规模异构机器人-视觉-语言数据集上预训练，用于通用迁移学习 | — | — |
| [`2toINF/X-VLA-AgiWorld-Challenge`](https://huggingface.co/2toINF/X-VLA-AgiWorld-Challenge) | Agibot-G1 | 针对AgiWorld挑战赛进行微调 | **冠军🥇** | - |
| [`2toINF/X-VLA-Calvin-ABC_D`](https://huggingface.co/2toINF/X-VLA-Calvin-ABC_D) | Franka机械臂 | 在CALVIN基准数据集（ABC_D子集）上微调 | **4.43** | [Calvin 评估指南](evaluation/calvin/README.md) |
| [`2toINF/X-VLA-Google-Robot`](https://huggingface.co/2toINF/X-VLA-Google-Robot) | Google机器人 | 在大规模谷歌机器人数据集上微调 | **83.5%(VM) 76.4%(VA)** | [Simpler 评估指南](evaluation/simpler/README.md) |
| [`2toINF/X-VLA-Libero`](https://huggingface.co/2toINF/X-VLA-Libero) | Franka机械臂 | 在LIBERO基准数据集上微调 | **98.1%** | [LIBERO 评估指南](evaluation/libero/README.md) |
| [`2toINF/X-VLA-VLABench`](https://huggingface.co/2toINF/X-VLA-VLABench) | Franka机械臂 | 在VLABench基准数据集上微调 | **51.1(分数)** | [VLABench 评估指南](evaluation/vlabench/README.md) |
| [`2toINF/X-VLA-RoboTwin2`](https://huggingface.co/2toINF/X-VLA-RoboTwin2) |  Agilex机器人 | 在RoboTwin2数据集上训练，实现双臂协同操作（每个任务仅使用50条演示数据） | **70%** | [RoboTwin2.0 评估指南](evaluation/robotwin-2.0/README.md) |
| [`2toINF/X-VLA-WidowX`](https://huggingface.co/2toINF/X-VLA-WidowX) | WidowX机械臂 | 在BridgeDataV2数据集（Simpler基准）上微调 | **95.8%** | [Simpler 评估指南](evaluation/simpler/README.md) |
| [`2toINF/X-VLA-SoftFold`](https://huggingface.co/2toINF/X-VLA-SoftFold) | Agilex机器人 | 在Soft-Fold数据集上微调，专攻柔性物体操作（如布料折叠、织物控制） | 两小时内布料折叠成功率达100% | [SoftFold-Agilex 评估指南](evaluation/SoftFold-Agilex/readme.md) |
| LoRA 适配权重 | | | | |
| [`2toINF/X-VLA-libero-spatial-peft`](https://huggingface.co/2toINF/X-VLA-libero-spatial-peft) | Franka机械臂 | 在LIBERO基准数据集上微调 | **96.2%** | [LIBERO 评估指南](evaluation/libero/README.md) |
| [`2toINF/X-VLA-libero-object-peft`](https://huggingface.co/2toINF/X-VLA-libero-object-peft) | Franka机械臂 | 在LIBERO基准数据集上微调 | **96%** | [LIBERO 评估指南](evaluation/libero/README.md) |
| [`2toINF/X-VLA-libero-goal-peft`](https://huggingface.co/2toINF/X-VLA-libero-goal-peft) | Franka机械臂 | 在LIBERO基准数据集上微调 | **94.4%** | [LIBERO 评估指南](evaluation/libero/README.md) |
| [`2toINF/X-VLA-libero-long-peft`](https://huggingface.co/2toINF/X-VLA-libero-long-peft) | Franka机械臂 | 在LIBERO基准数据集上微调 | **83.2%** | [LIBERO 评估指南](evaluation/libero/README.md) |
| [`2toINF/X-VLA-simpler-widowx-peft`](https://huggingface.co/2toINF/X-VLA-simpler-widowx-peft) | WidowX机械臂 | 在BridgeDataV2数据集（Simpler基准）上微调 | **66.7%** | [Simpler 评估指南](evaluation/simpler/README.md) |

---

## 🧩 备注说明
- 所有模型共用统一架构，相关文件包括`configuration_xvla.py`、`modeling_xvla.py`，以及统一的分词器文件（`tokenizer.json`）。
- **X-VLA-Pt** 是**基础预训练权重**，在多个机器人领域完成跨域预训练。
- 针对不同具身形态的微调，均在保留跨具身对齐能力的前提下，适配对应运行环境。
- 存放在`evaluation/`目录下的评估脚本，采用标准化格式，可复现基准测试结果。

---

> 📊 模型性能指标均遵循论文[《X-VLA》](https://arxiv.org/pdf/2510.10274)中详述的标准评估流程。

---

### 3️⃣ 启动推理服务端
```python
from transformers import AutoModel, AutoProcessor
import json_numpy

# 加载模型与处理器
model = AutoModel.from_pretrained("2toINF/X-VLA-WidowX", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("2toINF/X-VLA-WidowX", trust_remote_code=True)

# 启动推理服务
print("🚀 正在启动 X-VLA 推理服务...")
model.run(processor, host="0.0.0.0", port=8000)
```

服务启动后，API接口地址为：
```
POST http://<服务端IP>:8000/act
```

---

### 4️⃣ 客户端交互与动作预测
客户端通过HTTP POST请求与服务端通信，以JSON载荷的形式发送视觉、语言、本体感知等多模态数据。

#### 载荷数据结构
| 键名 | 数据类型 | 说明 |
| :--------------------- | :------------------------ | :---------------------------------------------------- |
| `proprio` | `json_numpy.dumps(array)` | 当前机器人本体感知状态（如关节角度） |
| `language_instruction` | `str` | 任务指令（例如：“拿起红色方块”） |
| `image0` | `json_numpy.dumps(array)` | 主相机采集的RGB图像 |
| `image1`、`image2` | *可选* | 其他视角相机图像（按需使用） |
| `domain_id` | `int` | 当前机器人具身形态/运行领域的标识 |
| `steps` | `int` | 基于流匹配生成方式的去噪步数（例如10步） |

#### 客户端示例代码
```python
import requests
import numpy as np
import json_numpy

server_url = "http://localhost:8000/act"
timeout = 5

# 准备输入数据
proprio = np.zeros(7, dtype=np.float32)
image = np.zeros((256, 256, 3), dtype=np.uint8)
instruction = "Move the gripper to the target position"
# 中文指令可替换为：“将夹持器移动至目标位置”

payload = {
    "proprio": json_numpy.dumps(proprio),
    "language_instruction": instruction,
    "image0": json_numpy.dumps(image),
    "domain_id": 0,
    "steps": 10
}

try:
    response = requests.post(server_url, json=payload, timeout=timeout)
    response.raise_for_status()
    result = response.json()
    actions = np.array(result["action"], dtype=np.float32)
    print(f"✅ 已接收 {actions.shape[0]} 组预测动作。")
except Exception as e:
    print(f"⚠️ 请求失败：{e}")
    actions = np.zeros((30, 20), dtype=np.float32)
```

#### 预期输出结果
```
[服务端] 模型已成功加载至 cuda:0
[服务端] 正在监听 0.0.0.0:8000 端口
[客户端] 正在向服务端发送观测数据...
✅ 已接收 30 组预测动作。
```

---

### 5️⃣ 标准化控制接口：EE6D
为实现不同具身形态的控制一致性，**X-VLA** 采用统一的**EE6D（末端执行器六维）**控制空间。

| 组成部分 | 规格说明 | 备注 |
| :------------------ | :------------------------------------------------------------------------- | :-------------------------------------------- |
| **本体感知输入** | 当前EE6D位姿（位置+姿态） | 需与训练阶段的数据归一化规则保持一致 |
| **动作输出** | 预测的目标相对/绝对位姿（EE6D格式） | 由下游控制器执行 |
| **向量维度** | 20维向量 = 3维（末端位置）+ 6维（六维旋转表征）+ 1维（夹持器状态）+ 10维（填充位） | |
| **单机械臂场景** | 若仅配置单臂，通过补零操作维持20维向量格式 | |

> ⚙️ **参考后处理代码：**
>
> ```python
> from datasets.utils import rotate6d_to_xyz
> action_final = np.concatenate([
>     action_pred[:3],
>     rotate6d_to_xyz(action_pred[3:9]),
>     np.array([1.0 if action_pred[9] > 0.5 else 0])
> ])
> ```
>
> 向模型输入本体感知数据时，需执行对应的**逆变换操作**。

---

### 6️⃣ 参考客户端实现
每一个开源模型，都在[`evaluation/<领域>/<机器人>/client.py`](evaluation/)路径下提供了对应的**参考客户端代码**，可复现标准部署行为。
在连接实体机器人或仿真环境时，我们强烈建议基于这些客户端代码进行适配改造。

---

### 7️⃣ SLURM 集群部署
针对大规模、分布式训练与部署场景（如高性能计算集群、AgiBot节点），可执行以下命令：
```bash
python -m deploy --model_path /path/to/your/model
```
该脚本会自动识别SLURM环境变量，启动分布式服务，并将连接相关元数据写入`info.json`文件。

---

## ⚙️ 自定义数据集的训练与微调
X-VLA 支持通过模块化、可扩展的数据集接口，在新的演示数据上开展微调训练。

### 数据准备流程
1.  **准备元数据JSON文件**：每个领域对应一个`meta.json`文件，记录轨迹文件路径。
2.  **实现自定义数据处理器**：编写领域加载类，实现`iter_episode(traj_idx)`生成器。
3.  **注册新领域**：修改以下文件：
    *   `datasets/domain_handler/registry.py`
    *   `datasets/domain_config.py`

### 示例数据处理器
| 处理器名称 | 适用数据集 | 说明 |
| :------------ | :-------------------- | :---------------------------------------- |
| `"lerobot"` | Agibot-Beta | 针对LEROBOT数据格式做了优化 |
| `"h5py"` | RoboMind / 仿真数据集 | 高效加载`.h5`格式的轨迹数据 |
| `"scattered"` | AGIWorld | 适配分散存储的轨迹数据 |

---

### 使用Accelerate启动训练
```bash
accelerate launch \
    --mixed_precision bf16 \
    train.py \
    --models '2toINF/X-VLA-Pt' \
    --train_metas_path /root/gpufree-data/libero_object_no_noops_lerobot_v21/meta/info.json \
    --learning_rate 1e-4 \
    --learning_coef 0.1 \
    --iters 50000 \
    --freeze_steps 1000 \
    --warmup_steps 2000

# 或者
./train.sh
```

| 参数 | 说明 |
| :------------------- | :------------------------------------- |
| `--models` | 基础模型（例如`'2toINF/X-VLA-Pt'`） |
| `--train_metas_path` | 元数据JSON文件路径 |
| `--batch_size` | 批次大小 |
| `--learning_rate` | 基础学习率 |
| `--learning_coef` | 软提示模块的学习率系数 |
| `--iters` | 总训练迭代次数 |
| `--freeze_steps` | 主干网络冻结的迭代步数 |
| `--warmup_steps` | 学习率预热迭代步数 |

---

## 📚 引用格式
如果你的研究工作使用了X-VLA，请按照以下格式引用：
```bibtex
@article{zheng2025x,
  title   = {X-VLA: Soft-Prompted Transformer as Scalable Cross-Embodiment Vision-Language-Action Model},
  author  = {Zheng, Jinliang and Li, Jianxiong and Wang, Zhihao and Liu, Dongxiu and Kang, Xirui
             and Feng, Yuchun and Zheng, Yinan and Zou, Jiayin and Chen, Yilun and Zeng, Jia and others},
  journal = {arXiv preprint arXiv:2510.10274},
  year    = {2025}
}
```

---

## 🪪 开源协议
本代码仓库采用 **Apache License 2.0** 开源协议。
在遵守协议条款的前提下，你可以自由使用、修改和分发本项目代码。

```
Copyright 2025 2toINF (https://github.com/2toinf)
Licensed under the Apache License, Version 2.0.
```

---

**由 [2toINF](https://github.com/2toinf) 团队维护**
💬 欢迎通过GitHub Discussions或提交Pull Request，反馈问题、提交意见与贡献代码。




## 🪪 huggingface
```bash
printenv | grep -E '^(HF|HUGGING)'

export HF_HOME="/root/gpufree-data/huggingface"
export HUGGINGFACE_HUB_CACHE="/root/gpufree-data/huggingface/hub"
export HF_ENDPOINT="https://hf-mirror.com"


# 安装hdf5-tools（首次使用需安装）
sudo apt install hdf5-tools

# 查看h5文件的顶层key
h5ls your_file.h5

# 递归查看所有key（包括子目录，最全面）
h5ls -r your_file.h5

# 查看指定key的详细信息（如形状、数据类型）
h5ls -v your_file.h5/abs_action_6d
```


## 数据及
- libero_goal_16包含16条轨迹的样本，可以用于训练模型
- 更多的数据可以[此处](https://huggingface.co/datasets/2toINF/Libero-XVLA-format)下载
- 在libero_goal_16/episodes.jsonl添加新增加数据的路径