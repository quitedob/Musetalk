# d:\python\Musetalk\dataset_processor\core\validator.py
"""
数据集质量验证器

验证数据集元数据的完整性和质量。
"""

import json
from pathlib import Path
from typing import Dict, List, Any


class DatasetValidator:
    """通用质量验证器"""

    def validate_meta_dir(self, meta_dir: str) -> Dict[str, List[str]]:
        """
        验证元数据目录
        
        Args:
            meta_dir: 元数据目录路径
            
        Returns:
            包含各类问题文件列表的字典
        """
        meta_path = Path(meta_dir)
        
        # 检查目录是否存在
        if not meta_path.exists():
            return {"error": [f"元数据目录不存在: {meta_dir}"]}
            
        issues: Dict[str, List[str]] = {
            "face_detection_failed": [],  # 人脸检测失败
            "audio_missing": [],          # 音频缺失
            "frame_mismatch": [],         # 帧数不一致或JSON错误
            "small_face": [],             # 人脸过小
            "invalid_json": [],           # JSON 解析失败
        }
        
        json_files = list(meta_path.glob("*.json"))
        if not json_files:
            issues["frame_mismatch"].append("未找到任何 .json 元数据文件")
            return issues
            
        for meta_file in json_files:
            try:
                data = json.loads(meta_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError as e:
                issues["invalid_json"].append(f"{meta_file.name}: {str(e)}")
                continue
            except Exception as e:
                issues["frame_mismatch"].append(f"{meta_file.name}: {str(e)}")
                continue
                
            # 检查有效性标记
            if not data.get("isvalid", True):
                issues["face_detection_failed"].append(meta_file.name)
                
            # 检查音频文件
            wav_path = data.get("wav_path", "")
            if wav_path:
                wav_full_path = Path(wav_path)
                # 尝试相对路径和绝对路径
                if not wav_full_path.is_absolute():
                    wav_full_path = meta_path.parent / wav_path
                if not wav_full_path.exists():
                    issues["audio_missing"].append(meta_file.name)
                    
            # 检查人脸尺寸 (带类型保护)
            face_size = data.get("face_size")
            if face_size is not None:
                # 确保 face_size 是列表/元组且有足够元素
                if isinstance(face_size, (list, tuple)) and len(face_size) >= 2:
                    try:
                        width, height = int(face_size[0]), int(face_size[1])
                        if width < 200 or height < 200:
                            issues["small_face"].append(meta_file.name)
                    except (ValueError, TypeError):
                        pass  # 无法转换为数字，跳过
                        
        return issues
    
    def validate_video_meta(self, meta_data: Dict[str, Any]) -> List[str]:
        """
        验证单个视频的元数据
        
        Args:
            meta_data: 元数据字典
            
        Returns:
            问题列表
        """
        issues = []
        
        # 必需字段检查
        required_fields = ["bbox", "frame_npaths"]
        for field in required_fields:
            if field not in meta_data:
                issues.append(f"缺少必需字段: {field}")
                
        # bbox 格式检查
        bbox = meta_data.get("bbox")
        if bbox is not None:
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                issues.append("bbox 格式无效，应为 [x1, y1, x2, y2]")
                
        # 帧路径检查
        frame_paths = meta_data.get("frame_npaths", [])
        if not frame_paths:
            issues.append("frame_npaths 为空")
            
        return issues
    
    def generate_report(self, issues: Dict[str, List[str]]) -> str:
        """
        生成验证报告
        
        Args:
            issues: validate_meta_dir 返回的问题字典
            
        Returns:
            格式化的报告字符串
        """
        lines = ["=" * 50, "数据集质量验证报告", "=" * 50, ""]
        
        total_issues = sum(len(v) for v in issues.values())
        lines.append(f"发现问题总数: {total_issues}")
        lines.append("")
        
        for category, files in issues.items():
            if files:
                lines.append(f"【{category}】({len(files)} 个)")
                for f in files[:10]:  # 最多显示10个
                    lines.append(f"  - {f}")
                if len(files) > 10:
                    lines.append(f"  ... 还有 {len(files) - 10} 个")
                lines.append("")
                
        return "\n".join(lines)
