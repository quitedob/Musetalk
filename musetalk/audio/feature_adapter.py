"""音频特征适配工具。"""

import math

import torch


def build_musetalk_audio_prompts(
    frame_features: torch.Tensor,
    librosa_length: int,
    fps: int = 25,
    audio_padding_length_left: int = 2,
    audio_padding_length_right: int = 2,
) -> torch.Tensor:
    """将帧级特征适配为 MuseTalk 所需的 [num_frames, 50, 384]。"""
    # 设定 MuseTalk 的内部音频帧率。  
    audio_fps = 50
    # 计算目标帧数量。  
    num_frames = math.floor((librosa_length / 16000.0) * int(fps))
    # 计算目标音频长度（50fps）。  
    target_audio_length = max(1, math.floor((librosa_length / 16000.0) * audio_fps))
    # 将输入特征重采样到 50fps 长度。  
    if frame_features.shape[0] != target_audio_length:
        resized = torch.nn.functional.interpolate(
            frame_features.T.unsqueeze(0),
            size=target_audio_length,
            mode="linear",
            align_corners=False,
        )
        frame_features = resized.squeeze(0).T.contiguous()

    # 计算每视频帧对应的音频窗口长度。  
    audio_feature_length_per_frame = 2 * (audio_padding_length_left + audio_padding_length_right + 1)
    # 计算索引倍率。  
    whisper_idx_multiplier = audio_fps / int(fps)
    # 计算前后 padding 大小。  
    padding_nums = math.ceil(whisper_idx_multiplier)

    # 添加首尾 padding，避免越界。  
    padded = torch.cat(
        [
            torch.zeros(
                padding_nums * audio_padding_length_left,
                frame_features.shape[1],
                device=frame_features.device,
                dtype=frame_features.dtype,
            ),
            frame_features,
            torch.zeros(
                padding_nums * 3 * audio_padding_length_right,
                frame_features.shape[1],
                device=frame_features.device,
                dtype=frame_features.dtype,
            ),
        ],
        dim=0,
    )

    # 按视频帧构建上下文窗口。  
    clips = []
    for frame_index in range(num_frames):
        # 计算当前帧中心对应音频索引。  
        audio_index = math.floor(frame_index * whisper_idx_multiplier)
        # 取固定长度窗口。  
        clip = padded[audio_index : audio_index + audio_feature_length_per_frame]
        # 边界场景下补齐窗口长度。  
        if clip.shape[0] < audio_feature_length_per_frame:
            clip = torch.cat(
                [
                    clip,
                    torch.zeros(
                        audio_feature_length_per_frame - clip.shape[0],
                        clip.shape[1],
                        device=clip.device,
                        dtype=clip.dtype,
                    ),
                ],
                dim=0,
            )
        # 模拟 Whisper 的 10x5 展开：每个时间步重复 5 次后拉平成 50。  
        clip = clip.unsqueeze(1).repeat(1, 5, 1).reshape(audio_feature_length_per_frame * 5, clip.shape[-1])
        clips.append(clip)

    # 拼接成最终张量。  
    if len(clips) == 0:
        return torch.zeros(0, 50, frame_features.shape[-1], device=frame_features.device, dtype=frame_features.dtype)
    return torch.stack(clips, dim=0)
