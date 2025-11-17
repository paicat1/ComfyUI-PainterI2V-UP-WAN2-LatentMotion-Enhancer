import torch
import comfy.model_management
import comfy.utils
import node_helpers
from torch.nn.functional import interpolate


# ==============================
# 原始单节点（功能完全保留）
# ==============================
class PainterI2V:
    """
    Wan2.2 图生视频增强节点 - 解决4步LoRA慢动作问题
    专为单帧输入优化，提升运动幅度，保持画面亮度稳定
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "width": ("INT", {"default": 832, "min": 16, "max": 4096, "step": 16}),
                "height": ("INT", {"default": 480, "min": 16, "max": 4096, "step": 16}),
                "length": ("INT", {"default": 81, "min": 1, "max": 4096, "step": 4}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                "motion_amplitude": ("FLOAT", {"default": 1.15, "min": 1.0, "max": 2.0, "step": 0.05}),
            },
            "optional": {
                "clip_vision_output": ("CLIP_VISION_OUTPUT",),
                "start_image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "execute"
    CATEGORY = "conditioning/video_models"

    def execute(self, positive, negative, vae, width, height, length, batch_size,
                motion_amplitude=1.15, start_image=None, clip_vision_output=None):
        # 1. 严格的零latent初始化（4步LoRA的生命线）
        latent = torch.zeros([batch_size, 16, ((length - 1) // 4) + 1, height // 8, width // 8], 
                           device=comfy.model_management.intermediate_device())
        
        if start_image is not None:
            # 单帧输入处理
            start_image = start_image[:1]
            start_image = comfy.utils.common_upscale(
                start_image.movedim(-1, 1), width, height, "bilinear", "center"
            ).movedim(1, -1)
            
            # 创建序列：首帧真实，后续0.5灰
            image = torch.ones((length, height, width, start_image.shape[-1]), 
                             device=start_image.device, dtype=start_image.dtype) * 0.5
            image[0] = start_image[0]
            
            concat_latent_image = vae.encode(image[:, :, :, :3])
            
            # 单帧mask：仅约束首帧
            mask = torch.ones((1, 1, latent.shape[2], concat_latent_image.shape[-2], 
                             concat_latent_image.shape[-1]), 
                            device=start_image.device, dtype=start_image.dtype)
            mask[:, :, 0] = 0.0
            
            # 2. 运动幅度增强（亮度保护核心算法）
            if motion_amplitude > 1.0:
                base_latent = concat_latent_image[:, :, 0:1]      # 首帧
                gray_latent = concat_latent_image[:, :, 1:]       # 灰帧
                
                diff = gray_latent - base_latent
                diff_mean = diff.mean(dim=(1, 3, 4), keepdim=True)
                diff_centered = diff - diff_mean
                scaled_latent = base_latent + diff_centered * motion_amplitude + diff_mean
                
                # Clamp & 组合
                scaled_latent = torch.clamp(scaled_latent, -6, 6)
                concat_latent_image = torch.cat([base_latent, scaled_latent], dim=2)
            
            # 3. 注入到conditioning
            positive = node_helpers.conditioning_set_values(
                positive, {"concat_latent_image": concat_latent_image, "concat_mask": mask}
            )
            negative = node_helpers.conditioning_set_values(
                negative, {"concat_latent_image": concat_latent_image, "concat_mask": mask}
            )

            # 4. 参考帧增强
            ref_latent = vae.encode(start_image[:, :, :, :3])
            positive = node_helpers.conditioning_set_values(positive, {"reference_latents": [ref_latent]}, append=True)
            negative = node_helpers.conditioning_set_values(negative, {"reference_latents": [torch.zeros_like(ref_latent)]}, append=True)

        if clip_vision_output is not None:
            positive = node_helpers.conditioning_set_values(positive, {"clip_vision_output": clip_vision_output})
            negative = node_helpers.conditioning_set_values(negative, {"clip_vision_output": clip_vision_output})

        out_latent = {"samples": latent}
        return (positive, negative, out_latent)


# ==============================
# 改进版节点（最终名称：PainterI2V-up WAN2.2 LatentMotion Enhancer by paicat1）
# ==============================
class PainterI2V_up_WAN2_2_LatentMotion_Enhancer_by_paicat1:
    """
    PainterI2V-up WAN2.2 LatentMotion Enhancer by paicat1（分区域幅度+动态模糊+扩展场景噪声）
    专为WAN2.2优化，潜空间运动强化核心，新增手动场景选择，绕开prompt提取问题，100%激活场景适配
    作者：paicat1（GitHub用户名）
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "width": ("INT", {"default": 832, "min": 16, "max": 4096, "step": 16}),
                "height": ("INT", {"default": 480, "min": 16, "max": 4096, "step": 16}),
                "length": ("INT", {"default": 81, "min": 1, "max": 4096, "step": 4}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                "motion_amplitude": ("FLOAT", {"default": 1.15, "min": 1.0, "max": 2.0, "step": 0.05}),
                # 新增：手动场景选择
                "manual_scene": ("COMBO", {
                    "default": "自动匹配",
                    "options": ["自动匹配", "动态主体（人物/动物）", "自然细节（风景/纹理）"]
                }),
                "noise_target": ("COMBO", {
                    "default": "全局（含环境）",
                    "options": ["全局（含环境）", "仅动作动态", "仅环境细节"]
                }),
                "noise_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.5, "step": 0.01}),
                "noise_decay_rate": ("FLOAT", {"default": 0.8, "min": 0.5, "max": 1.0, "step": 0.05}),
                "motion_threshold": ("FLOAT", {"default": 0.3, "min": 0.1, "max": 1.0, "step": 0.05}),
                "action_amplitude_boost": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 2.0, "step": 0.1}),
                "time_smooth_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.5, "step": 0.05}),
                "motion_blur_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.3, "step": 0.05}),
            },
            "optional": {
                "clip_vision_output": ("CLIP_VISION_OUTPUT",),
                "start_image": ("IMAGE",),
                "motion_mask": ("MASK",),
                "keyframe_image": ("IMAGE",),
                "keyframe_frame_idx": ("INT", {"default": 40, "min": 10, "max": 4096, "step": 1}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT", "LATENT")
    RETURN_NAMES = ("positive", "negative", "samples", "samples_2x_upscale")
    FUNCTION = "execute"
    CATEGORY = "Wan Video/Experimental"

    def execute(self, positive, negative, vae, width, height, length, batch_size,
                motion_amplitude=1.15, manual_scene="自动匹配", noise_target="全局（含环境）", 
                noise_strength=0.0, noise_decay_rate=0.8, motion_threshold=0.3, 
                action_amplitude_boost=1.0, time_smooth_strength=0.0, motion_blur_strength=0.0, 
                clip_vision_output=None, start_image=None, motion_mask=None, 
                keyframe_image=None, keyframe_frame_idx=40):
        
        if start_image is None:
            raise ValueError("【PainterI2V-up WAN2.2 LatentMotion Enhancer by paicat1】必须提供 start_image 输入")
        
        # 构建基础latent
        latent_t = ((length - 1) // 4) + 1
        latent_h, latent_w = height // 8, width // 8
        latent = torch.zeros(
            [batch_size, 16, latent_t, latent_h, latent_w],
            device=comfy.model_management.intermediate_device()
        )
        
        # 处理start_image（确保3通道）
        start_image = start_image[:1].squeeze(0)
        print(f"[通道调试] 初始形状: {start_image.shape}")
        if start_image.shape[-1] != 3:
            if start_image.shape[-1] == 1:
                start_image = start_image.repeat(1, 1, 3)
            else:
                start_image = start_image[..., :3]
        
        # 构建帧序列
        image_seq = torch.ones((length, height, width, 3), device=start_image.device, dtype=start_image.dtype) * 0.5
        image_seq[0] = start_image
        
        # VAE编码
        concat_latent_image = vae.encode(image_seq.unsqueeze(0))  # [1, C, T, H, W]
        print(f"[PainterI2V-up WAN2.2 LatentMotion Enhancer by paicat1] 📦 潜在空间形状: {concat_latent_image.shape}")
        
        # 构建mask
        mask = torch.ones((1, 1, latent_t, latent_h, latent_w), device=start_image.device, dtype=start_image.dtype)
        mask[:, :, 0] = 0.0
        
        # 对齐时间维度
        if concat_latent_image.shape[2] < latent_t:
            pad_length = latent_t - concat_latent_image.shape[2]
            concat_latent_image = torch.cat([
                concat_latent_image,
                concat_latent_image[:, :, -1:, :, :].repeat(1, 1, pad_length, 1, 1)
            ], dim=2)
            print(f"[维度对齐] 调整concat_latent_image时间维度至: {concat_latent_image.shape[2]}（与mask一致）")
        elif concat_latent_image.shape[2] > latent_t:
            concat_latent_image = concat_latent_image[:, :, :latent_t]

        # 判断是否启用高级功能
        enable_motion_enhance = motion_amplitude > 1.0
        enable_region_boost = enable_motion_enhance and (action_amplitude_boost > 1.0 or motion_mask is not None)
        enable_noise = noise_strength > 0.01
        enable_time_smooth = time_smooth_strength > 0.01
        enable_motion_blur = motion_blur_strength > 0.01

        spatial_motion_mask = None

        # 运动掩码生成
        if enable_region_boost or enable_noise or enable_motion_blur:
            if motion_mask is not None:
                # 补全通道维度
                if len(motion_mask.shape) == 3:
                    motion_mask = motion_mask.unsqueeze(1)
                elif len(motion_mask.shape) == 2:
                    motion_mask = motion_mask.unsqueeze(0).unsqueeze(0)
                
                # 缩放到latent尺寸
                spatial_motion_mask = interpolate(
                    motion_mask, 
                    size=(latent_h, latent_w),
                    mode='nearest'
                )
                
                # 扩展到时间维度
                spatial_motion_mask = spatial_motion_mask.repeat(1, 1, latent_t, 1, 1)
                print(f"[PainterI2V-up WAN2.2 LatentMotion Enhancer by paicat1] 🖌️ 自动处理后遮罩形状: {spatial_motion_mask.shape}")
            else:
                base_latent = concat_latent_image[:, :, 0:1]
                gray_latent = concat_latent_image[:, :, 1:]
                diff = torch.abs(gray_latent - base_latent.mean(dim=2, keepdim=True))
                motion_intensity = diff.mean(dim=1, keepdim=True)
                adaptive_threshold = max(0.1, min(motion_threshold, 1.0))
                smoothed_intensity = interpolate(motion_intensity, size=(latent_t, latent_h, latent_w), mode='trilinear')
                spatial_motion_mask = (smoothed_intensity > adaptive_threshold).float()
                spatial_motion_mask[:, :, 0] = 0.0
                print(f"[PainterI2V-up WAN2.2 LatentMotion Enhancer by paicat1] 🎯 掩码阈值: {adaptive_threshold} | 覆盖率: {spatial_motion_mask.mean().item()*100:.1f}%")

        # 运动幅度增强
        if enable_motion_enhance:
            base_latent = concat_latent_image[:, :, 0:1]
            gray_latent = concat_latent_image[:, :, 1:]
            diff = gray_latent - base_latent
            diff_mean = diff.mean(dim=(1, 3, 4), keepdim=True)
            diff_centered = diff - diff_mean

            # 分区域增强
            if enable_region_boost and spatial_motion_mask is not None:
                action_mask = spatial_motion_mask[:, :, 1:1+gray_latent.shape[2]]
                diff_centered = diff_centered * (1.0 + (action_amplitude_boost - 1.0) * action_mask)
                print(f"[PainterI2V-up WAN2.2 LatentMotion Enhancer by paicat1] 🚀 分区域运动增强生效 | 增强系数: {action_amplitude_boost}")

            # 应用幅度增强
            scaled_latent = base_latent + diff_centered * motion_amplitude + diff_mean
            scaled_latent = torch.clamp(scaled_latent, -6, 6)
            concat_latent_image = torch.cat([base_latent, scaled_latent], dim=2)
            print(f"[PainterI2V-up WAN2.2 LatentMotion Enhancer by paicat1] 📈 运动幅度增强完成 | 幅度: {motion_amplitude}")

            # 时间平滑
            if enable_time_smooth:
                for t in range(1, concat_latent_image.shape[2]):
                    concat_latent_image[:, :, t] = (
                        concat_latent_image[:, :, t-1] * time_smooth_strength +
                        concat_latent_image[:, :, t] * (1 - time_smooth_strength)
                    )
                print(f"[PainterI2V-up WAN2.2 LatentMotion Enhancer by paicat1] ⚡ 时间平滑生效 | 强度: {time_smooth_strength}")

        # 噪声注入
        if enable_noise:
            noise_base = torch.randn_like(concat_latent_image)
            low_freq = interpolate(noise_base, scale_factor=(1.0, 0.5, 0.5), mode='trilinear', align_corners=False)
            low_freq = interpolate(low_freq, size=noise_base.shape[2:], mode='trilinear', align_corners=False)
            high_freq = noise_base - low_freq

            # 场景关键词库
            SCENE_CATEGORIES = [
                {
                    "name": "dynamic_subject",
                    "keywords": [
                        "person", "people", "human", "man", "woman", "girl", "boy", "child", "face", "portrait",
                        "character", "creature", "animal", "dog", "cat", "bird", "monster", "robot", "figure", "body",
                        "人物", "人像", "角色", "生物", "动物", "人脸", "肖像", "女孩", "男孩", "小孩", "机器人", "身体", "模特"
                    ],
                    "low_ratio": 0.7,
                    "high_ratio": 0.3,
                },
                {
                    "name": "natural_detail",
                    "keywords": [
                        "nature", "forest", "tree", "leaf", "water", "fire", "smoke", "cloud", "sky", "mountain",
                        "particle", "dust", "spark", "rain", "snow", "fog", "mist", "light ray", "bokeh", "texture",
                        "grass", "flower", "ocean", "river", "storm", "explosion", "magic", "aura", "fluid", "wave",
                        "自然", "森林", "树叶", "水", "火", "烟", "云", "天空", "山脉", "粒子", "灰尘", "火花",
                        "雨", "雪", "雾", "光效", "景深", "纹理", "草地", "花朵", "海洋", "河流", "风暴", "爆炸", "魔法", "流体", "波浪"
                    ],
                    "low_ratio": 0.3,
                    "high_ratio": 0.7,
                }
            ]

            # 手动场景优先适配
            dynamic_noise = None
            if manual_scene == "动态主体（人物/动物）":
                matched_category = SCENE_CATEGORIES[0]
                dynamic_noise = low_freq * matched_category["low_ratio"] + high_freq * matched_category["high_ratio"]
                print(f"[噪声适配] 手动选择场景: {matched_category['name']}类 | 低频占比: {matched_category['low_ratio']}")
            elif manual_scene == "自然细节（风景/纹理）":
                matched_category = SCENE_CATEGORIES[1]
                dynamic_noise = low_freq * matched_category["low_ratio"] + high_freq * matched_category["high_ratio"]
                print(f"[噪声适配] 手动选择场景: {matched_category['name']}类 | 低频占比: {matched_category['low_ratio']}")
            else:
                # 自动匹配逻辑
                prompt_text = ""
                for cond in positive:
                    if isinstance(cond, (list, tuple)) and len(cond) > 1:
                        meta = cond[1]
                        if isinstance(meta, dict) and "text" in meta:
                            prompt_text += meta["text"].lower() + " "
                print(f"[调试] 自动匹配模式 - 提取到的prompt: {prompt_text}")

                if not prompt_text.strip():
                    dynamic_noise = (low_freq + high_freq) * 0.5
                    print(f"[噪声适配] 自动匹配 - 未匹配特定场景，使用默认比例")
                else:
                    matched_category = None
                    for category in SCENE_CATEGORIES:
                        if any(kw in prompt_text for kw in category["keywords"]):
                            matched_category = category
                            break
                    if matched_category:
                        dynamic_noise = low_freq * matched_category["low_ratio"] + high_freq * matched_category["high_ratio"]
                        print(f"[噪声适配] 自动匹配 - 匹配场景: {matched_category['name']}类 | 低频占比: {matched_category['low_ratio']}")
                    else:
                        dynamic_noise = (low_freq + high_freq) * 0.5
                        print(f"[噪声适配] 自动匹配 - 未匹配特定场景，使用默认比例")

            # 应用噪声目标区域
            if noise_target == "仅动作动态" and spatial_motion_mask is not None:
                if spatial_motion_mask.shape[2] != dynamic_noise.shape[2]:
                    spatial_motion_mask = interpolate(spatial_motion_mask, size=(dynamic_noise.shape[2], latent_h, latent_w), mode='nearest')
                dynamic_noise = dynamic_noise * spatial_motion_mask
            elif noise_target == "仅环境细节" and spatial_motion_mask is not None:
                if spatial_motion_mask.shape[2] != dynamic_noise.shape[2]:
                    spatial_motion_mask = interpolate(spatial_motion_mask, size=(dynamic_noise.shape[2], latent_h, latent_w), mode='nearest')
                dynamic_noise = dynamic_noise * (1.0 - spatial_motion_mask)

            # 时间衰减
            time_weights = torch.linspace(1.0, noise_decay_rate, dynamic_noise.shape[2], device=dynamic_noise.device)
            dynamic_noise = dynamic_noise * time_weights.view(1, 1, -1, 1, 1)
            
            # 注入并裁剪
            concat_latent_image = concat_latent_image + dynamic_noise * noise_strength
            concat_latent_image = torch.clamp(concat_latent_image, -6, 6)
            print(f"[PainterI2V-up WAN2.2 LatentMotion Enhancer by paicat1] ✅ 噪声注入生效 | 目标: {noise_target} | 强度: {noise_strength}")

        # 动态模糊
        if enable_motion_blur and spatial_motion_mask is not None:
            blurred = concat_latent_image.clone()
            for t in range(1, concat_latent_image.shape[2]):
                region = spatial_motion_mask[:, :, t] > 0.5
                if region.any():
                    blurred[:, :, t, region.squeeze(0).squeeze(0)] = (
                        concat_latent_image[:, :, t-1, region.squeeze(0).squeeze(0)] * motion_blur_strength +
                        concat_latent_image[:, :, t, region.squeeze(0).squeeze(0)] * (1 - motion_blur_strength)
                    )
            concat_latent_image = blurred
            print(f"[PainterI2V-up WAN2.2 LatentMotion Enhancer by paicat1] 🌪️ 动态模糊生效 | 强度: {motion_blur_strength}")

        # 设置Conditioning
        positive = node_helpers.conditioning_set_values(positive, {"concat_latent_image": concat_latent_image, "concat_mask": mask})
        negative = node_helpers.conditioning_set_values(negative, {"concat_latent_image": concat_latent_image, "concat_mask": mask})

        print(f"[通道调试] 编码ref前形状: {start_image.unsqueeze(0).shape}")
        ref_latent = vae.encode(start_image.unsqueeze(0)[:, :, :, :3])
        positive = node_helpers.conditioning_set_values(positive, {"reference_latents": [ref_latent]}, append=True)
        negative = node_helpers.conditioning_set_values(negative, {"reference_latents": [torch.zeros_like(ref_latent)]}, append=True)

        if clip_vision_output is not None:
            positive = node_helpers.conditioning_set_values(positive, {"clip_vision_output": clip_vision_output})
            negative = node_helpers.conditioning_set_values(negative, {"clip_vision_output": clip_vision_output})

        # 输出构建
        samples = {"samples": latent, "batch_size": batch_size, "frame_rate": 24}
        B, C, T, H, W = latent.shape
        latent_4d = latent.reshape(B * T, C, H, W)
        upscaled_4d = interpolate(latent_4d, size=(H*2, W*2), mode='bilinear', align_corners=False)
        upscaled_latent = upscaled_4d.reshape(B, C, T, H*2, W*2)
        samples_2x_upscale = {"samples": upscaled_latent, "batch_size": batch_size, "frame_rate": 24}
        print(f"[PainterI2V-up WAN2.2 LatentMotion Enhancer by paicat1] 📤 输出完成 | 原尺寸: {latent.shape} | 2倍放大: {upscaled_latent.shape}")

        return (positive, negative, samples, samples_2x_upscale)


# ==============================
# 节点注册（确保ComfyUI正常识别）
# ==============================
NODE_CLASS_MAPPINGS = {
    "PainterI2V": PainterI2V,
    "PainterI2V-up WAN2.2 LatentMotion Enhancer by paicat1": PainterI2V_up_WAN2_2_LatentMotion_Enhancer_by_paicat1
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PainterI2V": "🎨 PainterI2V (Wan2.2 慢动作修复)",
    "PainterI2V-up WAN2.2 LatentMotion Enhancer by paicat1": "🚀 PainterI2V-up WAN2.2 LatentMotion Enhancer by paicat1（潜空间运动强化）"
}
