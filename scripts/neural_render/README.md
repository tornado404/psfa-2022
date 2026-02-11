# 神经渲染脚本 (Neural Render Script)

本目录包含用于 3D 人脸模型神经渲染的脚本，利用所提出的神经渲染方法实现人脸重演（Face Reenactment）和动画生成。

## 用法 (Usage)

您可以从项目根目录通过命令行运行神经渲染脚本。

### 命令 (Command)

```bash
python -m scripts.neural_render \
    --out_path <输出视频路径> \
    --iden_path <身份网格路径> \
    --offsets_npy <偏移量numpy文件> \
    --reenact_video <参考视频路径> \
    --reenact_coeff <重演系数路径> \
    --nr_ckpt <神经渲染器检查点> \
    --audio_path <音频文件路径> \
    [--reenact_static_frame <帧索引>]
```

### 参数 (Arguments)

| 参数名 | 类型 | 必填 | 描述 |
|--------|------|------|------|
| `--out_path` | str | 是 | 生成的输出视频保存路径。 |
| `--iden_path` | str | 是 | 身份网格文件路径（例如 `.obj`）。 |
| `--offsets_npy` | str | 是 | 包含顶点偏移量的 `.npy` 文件路径。 |
| `--reenact_video` | str | 是 | 用作重演参考的视频路径。 |
| `--reenact_coeff` | str | 是 | 包含重演系数/数据的路径或目录。 |
| `--nr_ckpt` | str | 是 | 神经渲染器模型检查点路径。 |
| `--audio_path` | str | 是 | 与视频同步的音频文件路径。 |
| `--reenact_static_frame` | int | 否 | 如果使用静态帧进行重演，则指定帧索引（可选）。 |

### 示例 (Example)

```bash
python -m scripts.neural_render \
    --out_path output/result.mp4 \
    --iden_path assets/identity.obj \
    --offsets_npy assets/offsets.npy \
    --reenact_video data/video.mp4 \
    --reenact_coeff data/coeffs \
    --nr_ckpt checkpoints/renderer.pth \
    --audio_path data/audio.wav
```

## 详细说明 (Description)

该脚本加载基础身份网格（Identity Mesh）并应用顶点偏移（Vertex Offsets）对其进行动画处理。然后，它使用神经渲染器根据变形后的网格和重演信号生成逼真的视频帧，并与提供的音频同步。
