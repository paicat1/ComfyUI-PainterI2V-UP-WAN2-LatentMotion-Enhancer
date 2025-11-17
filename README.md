# PainterI2V-up WAN2.2 LatentMotion Enhancer by paicat1

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![ComfyUI Compatible](https://img.shields.io/badge/ComfyUI-Compatible-green.svg)](https://github.com/comfyanonymous/ComfyUI)

> 专为 WAN2.2 优化的图生视频潜空间运动强化节点，基于动画小子大佬的 PainterI2V 核心思路深化开发，提供更精细的动态效果控制。

## 项目简介
本节点受到动画小子大佬（GitHub：[princepainter](https://github.com/princepainter)）开发的 [PainterI2V](https://github.com/princepainter/ComfyUI-PainterI2V) 节点启发，是其后续优化版本。

动画小子大佬开创性地实现了**潜空间帧间动态强化**，有效解决了 WAN2.2 模型配合 4 步 LoRA（如 lightx2v）时的慢动作问题。本节点在此基础上，进一步探索潜空间运动优化的可能性，新增分区域增强、场景自适应噪声、动态模糊等功能，让图生视频的动态效果更自然、更可控。

- 作者：paicat1（GitHub：[paicat1](https://github.com/paicat1)）
- 原项目：[ComfyUI-PainterI2V](https://github.com/princepainter/ComfyUI-PainterI2V)（感谢动画小子大佬的开源贡献）

## 核心亮点
- ✅ **WAN2.2 深度适配**：完美兼容 WAN2.2 图生视频工作流，无版本兼容问题
- ✅ **潜空间精准调控**：基于 `LatentMotion` 技术，直接在低维潜空间优化运动信息
- ✅ **多维度动态增强**：支持分区域运动放大、场景自适应噪声、帧间平滑与动态模糊
- ✅ **场景智能匹配**：手动/自动切换场景类型（动态主体/自然细节），绕开 prompt 提取误差
- ✅ **双分辨率输出**：同时提供原始尺寸与 2 倍放大 latent，适配不同生成需求

## 安装指南
### 前置依赖
- 已安装 [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- 已部署 WAN2.2 图生视频模型及配套 VAE
- Python 3.8+

### 安装步骤
1. 克隆本仓库（或直接下载代码文件）：
   ```bash
   git clone https://github.com/paicat1/ComfyUI-PainterI2V-up-WAN2.2-LatentMotion-Enhancer.git
   ```
2. 将节点文件 `PainterI2V-up_WAN2.2_LatentMotion_Enhancer_by_paicat1.py` 复制到 ComfyUI 的 `custom_nodes` 目录下
3. 重启 ComfyUI，节点将自动加载至 `Wan Video/Experimental` 分类中

## 使用示例
### 基础工作流
1. 准备输入资源：
   - `start_image`：图生视频的起始帧（必填）
   - `VAE`：与 WAN2.2 配套的 VAE 模型
   - `positive/negative`：提示词条件节点
2. 配置节点参数：
   - 基础参数：设置视频宽高（如 832x480）、帧数（如 81）、批量大小
   - 核心参数：`motion_amplitude=1.2`（运动幅度）、`manual_scene=动态主体`（场景类型）
   - 优化参数：`noise_strength=0.1`（噪声强度）、`motion_blur_strength=0.05`（动态模糊）
3. 连接生成节点：
   - 将节点输出的 `positive`/`negative` 连接到 WAN2.2 生成节点
   - 选择 `samples`（原始尺寸）或 `samples_2x_upscale`（2 倍放大）作为 latent 输入
4. 运行工作流，生成视频

### 场景参数推荐
| 场景类型 | 推荐参数组合 | 效果说明 |
|----------|--------------|----------|
| 人物/动物动画 | `manual_scene=动态主体` + `motion_amplitude=1.2-1.3` + `action_amplitude_boost=1.3` | 强化主体运动幅度，保持肢体连贯 |
| 风景/纹理动态 | `manual_scene=自然细节` + `noise_strength=0.15-0.2` + `noise_target=仅环境细节` | 增强环境动态细节（如烟雾、水流） |
| 通用场景 | `manual_scene=自动匹配` + `motion_amplitude=1.1` + `time_smooth_strength=0.2` | 平衡运动感与画面稳定性 |

### 工作流截图（占位）
![工作流示例](https://via.placeholder.com/800x450?text=ComfyUI+Workflow+Screenshot)  
*（请替换为实际工作流截图，建议展示完整的输入-配置-输出连接关系）*

## 参数速览
| 参数分类 | 核心参数 | 默认值 | 推荐范围 |
|----------|----------|--------|----------|
| 基础配置 | width/height | 832/480 | 16-4096（步长16） |
| 基础配置 | length | 81 | 1-4096（步长4） |
| 运动控制 | motion_amplitude | 1.15 | 1.0-1.3 |
| 运动控制 | action_amplitude_boost | 1.0 | 1.0-1.5 |
| 场景适配 | manual_scene | 自动匹配 | 动态主体/自然细节/自动匹配 |
| 噪声优化 | noise_strength | 0.0 | 0.05-0.2 |
| 平滑优化 | time_smooth_strength | 0.0 | 0.1-0.3 |
| 模糊优化 | motion_blur_strength | 0.0 | 0.05-0.15 |

完整参数说明见 [详细文档](docs/参数说明.md)（可选：如需拆分详细文档可新增此目录）

## 注意事项
1. **必填参数**：`start_image` 为核心输入，未提供将直接报错
2. **模型兼容性**：仅适配 WAN2.2 模型，其他版本可能出现运动异常
3. 显存占用：高分辨率（如 1024x768）+ 高帧数（如 200+）建议使用 12GB+ 显存显卡
4. 调参原则：避免同时调高 `motion_amplitude` 和 `noise_strength`，防止画面混乱
5. 日志查看：运行时可通过 ComfyUI 控制台查看节点输出日志，辅助调参

## 许可证
本项目基于 [MIT 许可证](LICENSE) 开源，使用时需保留原作者（动画小子 princepainter、后续开发 paicat1）署名。

- 可自由使用、修改、分发本节点
- 禁止用于商业用途时移除作者署名
- 作者不对使用本节点产生的任何损失承担责任

## 贡献指南
欢迎通过以下方式参与项目贡献：
1. Fork 本仓库
2. 创建特性分支（`git checkout -b feature/xxx`）
3. 提交修改（`git commit -m 'Add xxx feature'`）
4. 推送分支（`git push origin feature/xxx`）
5. 发起 Pull Request

贡献方向：
- 新增场景适配类型
- 优化运动控制算法
- 降低显存占用
- 补充多语言文档

## 问题反馈
如遇到使用问题或功能建议，可通过以下方式反馈：
1. GitHub Issues：[提交 Issue](https://github.com/paicat1/ComfyUI-PainterI2V-up-WAN2.2-LatentMotion-Enhancer/issues)
2. 联系作者：通过 GitHub 主页留言

反馈时建议提供：
- 完整的报错日志
- 工作流截图
- 所用参数配置
- 显卡型号与显存大小

---

如果觉得这个节点对你有帮助，欢迎点亮 ⭐️ Star 支持一下～
