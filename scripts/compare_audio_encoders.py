"""音频编码器性能对比脚本。"""

import argparse
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from musetalk.audio.campplus_encoder import CamPlusAudioEncoder
from musetalk.audio.sensevoice_encoder import SenseVoiceAudioEncoder
from musetalk.audio.whisper_encoder import WhisperAudioEncoder


class AudioEncoderBenchmark:
    """音频编码器基准测试器。"""

    def __init__(self, test_audio_path: str, device: str = "cuda") -> None:
        # 保存测试音频路径。  
        self.test_audio_path = test_audio_path
        # 保存运行设备。  
        self.device = device

    def benchmark_encoder(self, encoder_ctor, encoder_name: str, num_runs: int = 5) -> Dict:
        """测试单个编码器的时延、显存和输出形状。"""
        # 构建编码器实例。  
        encoder = encoder_ctor()
        # 记录参数量。  
        param_count = encoder.param_count

        # 在 CUDA 下统计峰值显存。  
        if torch.cuda.is_available() and "cuda" in self.device:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            base_mem = torch.cuda.memory_allocated() / 1024**2
        else:
            base_mem = 0.0

        # 多次运行统计平均时延。  
        times = []
        output_shape = None
        for _ in range(num_runs):
            if torch.cuda.is_available() and "cuda" in self.device:
                torch.cuda.synchronize()
            start_t = time.time()
            features, _ = encoder.encode(self.test_audio_path)
            if torch.cuda.is_available() and "cuda" in self.device:
                torch.cuda.synchronize()
            end_t = time.time()
            times.append((end_t - start_t) * 1000.0)
            output_shape = tuple(features.shape)

        # 读取峰值显存。  
        if torch.cuda.is_available() and "cuda" in self.device:
            peak_mem = torch.cuda.max_memory_allocated() / 1024**2
            memory_mb = max(0.0, peak_mem - base_mem)
        else:
            memory_mb = 0.0

        return {
            "encoder": encoder_name,
            "param_count": int(param_count),
            "avg_time_ms": float(np.mean(times)),
            "std_time_ms": float(np.std(times)),
            "peak_memory_mb": float(memory_mb),
            "feature_shape": str(output_shape),
        }

    def run(self, encoders: List[Tuple], num_runs: int = 5) -> pd.DataFrame:
        """执行全部编码器测试并返回结果表。"""
        rows = []
        for encoder_ctor, encoder_name in encoders:
            try:
                rows.append(self.benchmark_encoder(encoder_ctor, encoder_name, num_runs=num_runs))
            except Exception as exc:
                rows.append(
                    {
                        "encoder": encoder_name,
                        "param_count": -1,
                        "avg_time_ms": -1,
                        "std_time_ms": -1,
                        "peak_memory_mb": -1,
                        "feature_shape": f"ERROR: {exc}",
                    }
                )
        return pd.DataFrame(rows)


def main() -> None:
    """脚本入口。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_path", type=str, required=True, help="测试音频路径")
    parser.add_argument("--device", type=str, default="cuda", help="运行设备")
    parser.add_argument("--num_runs", type=int, default=5, help="每个编码器重复次数")
    parser.add_argument("--save_csv", type=str, default="", help="可选结果输出 CSV 路径")
    args = parser.parse_args()

    benchmark = AudioEncoderBenchmark(test_audio_path=args.audio_path, device=args.device)
    encoders = [
        (lambda: WhisperAudioEncoder(model_path="./models/whisper", device=args.device), "whisper"),
        (lambda: CamPlusAudioEncoder(device=args.device), "campplus"),
        (lambda: SenseVoiceAudioEncoder(device=args.device), "sensevoice"),
    ]
    result_df = benchmark.run(encoders=encoders, num_runs=args.num_runs)
    print(result_df.to_string(index=False))
    if args.save_csv:
        result_df.to_csv(args.save_csv, index=False)
        print(f"Saved benchmark CSV: {args.save_csv}")


if __name__ == "__main__":
    main()
