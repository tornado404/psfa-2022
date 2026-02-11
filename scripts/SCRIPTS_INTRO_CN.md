# 脚本使用说明文档

本文档详细介绍了 `scripts/` 目录下用于数字人推理生成的两个核心脚本：`generate_animnet.sh` 和 `generate_rendering.sh`。

这两个脚本通常需要按顺序执行（先生成动作，再进行渲染），共同完成从"输入音频"到"输出说话视频"的全过程。

---

## 1. generate_animnet.sh (动作生成脚本)

### 功能简介
该脚本主要负责**动画推理**阶段。它调用 `GenAnimNet` 程序，根据输入的驱动音频，预测出人脸的运动偏移量（Offsets）。这是生成过程的第一步。

### 命令格式
```bash
./scripts/generate_animnet.sh <SPEAKER_ID> <AUDIO_PATH> <TASK_ID>
```

### 参数详解
| 参数位置 | 参数名 | 含义 | 示例 |
| :--- | :--- | :--- | :--- |
| **$1** | `SPEAKER_ID` | 说话人/角色 ID，用于指定加载哪个角色的模型配置。 | `obama`, `biden` |
| **$2** | `AUDIO_PATH` | 驱动音频文件的绝对路径（支持 wav 等格式）。 | `/path/to/input.wav` |
| **$3** | `TASK_ID` | 任务唯一标识符，用于隔离不同任务的输出目录。 | `task_12345` |

### 核心逻辑说明
1. **环境准备**：自动检测并激活 `psfa` Conda 环境，设置 `PROJECT_ROOT`。
2. **执行推理**：
   - 运行 `GenAnimNet` 命令。
   - 使用 `--generating` 标志开启生成模式。
   - 通过 `--test_media` 和 `model.test_media.misc` 参数动态注入音频路径和任务 ID，覆盖配置文件中的默认值。
   - 使用 `--dump_offsets` 标志将预测结果保存为 `.npy` 文件。
3. **输出检查**：脚本最后会检查是否成功生成了关键文件。

### 输出产物
生成的文件通常位于 `runs/anime/<SPEAKER_ID>/animnet-decmp/generated/` 目录下包含 `TASK_ID` 的子文件夹中：
- `dump-offsets-final.npy`: 预测得到的人脸动作偏移量数据（下一步渲染的输入）。
- `audio.wav`: 预处理后的音频文件。

---

## 2. generate_rendering.sh (神经渲染脚本)

### 功能简介
该脚本负责**视频渲染**阶段。它使用上一步生成的动作偏移量（Offsets）和神经渲染器（Neural Renderer），将抽象的动作数据合成为逼真的 MP4 视频。

### 命令格式
```bash
./scripts/generate_rendering.sh <SPEAKER_ID> <AUDIO_PATH> <TASK_ID>
```

### 参数详解
| 参数位置 | 参数名 | 含义 | 示例 |
| :--- | :--- | :--- | :--- |
| **$1** | `SPEAKER_ID` | 说话人/角色 ID（必须与第一步保持一致）。 | `obama`, `biden` |
| **$2** | `AUDIO_PATH` | 输入音频路径（脚本主要使用上一步生成的 `audio.wav`，保留此参数是为了保持接口统一）。 | `/path/to/input.wav` |
| **$3** | `TASK_ID` | 任务唯一标识符，用于查找第一步生成的中间文件。 | `task_12345` |

### 核心逻辑说明
1. **查找输入**：根据 `TASK_ID` 在 `runs/anime/.../generated` 目录下搜索包含动作数据的文件夹。
2. **执行渲染**：
   - 调用 `python -m scripts.neural_render`。
   - 加载神经渲染器权重 (`--nr_ckpt`)。
   - 读取身份模型 (`--iden_path`) 和重演视频参考 (`--reenact_video`)。
   - 将动作数据 (`--offsets_npy`) 和音频 (`--audio_path`) 结合生成视频。
3. **输出结果**：生成最终视频文件。

### 输出产物
- `output.mp4`: 最终合成的数字人说话视频，位于中间文件目录下。

---

## 4. train_animnet.sh (动作生成网络训练脚本)

### 功能简介
该脚本用于训练特定人物的 AnimNet 模型（对应 `functions.sh` 中的 `TrainAnimNetDecmp`）。它学习输入音频与该人物面部动作之间的映射关系（即人物风格）。训练产物将用于后续的 `generate_animnet.sh` 脚本。

### 命令格式
```bash
./scripts/train_animnet.sh <SPEAKER_ID> [DATA_SRC]
```

### 参数详解
| 参数位置 | 参数名 | 含义 | 默认值/示例 |
| :--- | :--- | :--- | :--- |
| **$1** | `SPEAKER_ID` | 目标训练人物 ID | `m001_trump` |
| **$2** | `DATA_SRC` | 数据源名称 | `celebtalk` |

### 输出产物
e.g. `./scripts/train_animnet.sh m001_trump celebtalk`
训练过程会生成检查点文件（checkpoints），通常保存在 `runs/anime/<SPEAKER_ID>/animnet-decmp/checkpoints/` 目录下。

---

## 5. 注意事项

1. **硬编码路径**：
   脚本中定义了项目根目录变量 `PROJECT_ROOT="/mnt/d/Downloads/psfa-2022-main"`。如果项目位置发生变动，**必须**修改这两个脚本中的此变量。

2. **Conda 环境**：
   脚本会自动尝试在常见位置（如 `~/miniconda3`, `/home/sdk/anaconda3` 等）查找 Conda 并激活 `psfa` 环境。请确保您的环境中已创建名为 `psfa` 的虚拟环境。

3. **资源依赖**：
   脚本运行依赖于 `assets/` 目录下的身份模型数据和 `runs/` 目录下的预训练模型权重。请确保这些资源文件完整存在。

---

## 3. functions.sh (通用功能脚本)

### 功能简介
该脚本是一个 Bash 函数库，被其他脚本（如 `generate.sh`, `generate_animnet.sh` 等）引用（source）。它定义了参数解析函数 `GetArgs` 以及一系列训练和生成函数（如 `TrainAnimNet`, `GenAnimNet` 等），用于简化对 Python 主程序 (`src`) 的调用。

### 核心函数

#### GetArgs
解析命令行参数并将其转换为 Hydra 配置格式。

#### GenAnimNet
封装了 `python3 -m src mode=generate` 命令，用于执行推理生成。

### 常用参数 (传递给 GetArgs)

| 参数名 | 含义 | 默认值/示例 |
| :--- | :--- | :--- |
| `--exp` | 实验类型 (Experiment type) | `decmp`, `track`, `cmb3d` |
| `--exp_name` | 实验名称 (Experiment name) | `animnet-decmp` |
| `--data_src` | 数据源 (Data source) | `celebtalk` |
| `--speaker` | 说话人 ID (Speaker ID) | `obama`, `m001_trump` |
| `--generating` | 开启生成模式标志 | (无值) |
| `--load` | 加载的模型检查点文件名 (用于 GenAnimNet) | `epoch_50.pth` |
| `--test_media` | 测试媒体路径/ID | `gen_obama` |
| `--dump_offsets` | 是否保存 offsets 数据 | (无值) |

### 用法示例
通常不需要直接运行此脚本，而是在其他脚本中引用它：

```bash
source scripts/functions.sh

# 调用 GenAnimNet 函数进行生成
GenAnimNet --generating \
  --exp=decmp \
  --exp_name=animnet-decmp \
  --load='epoch_50.pth' \
  --speaker="obama" \
  ...
```

