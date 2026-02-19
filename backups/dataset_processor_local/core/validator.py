# f:\python\Musetalk\dataset_processor\core\validator.py
import json  # 导入 JSON 模块
from pathlib import Path  # 导入路径模块
from typing import Dict, List  # 导入类型注解


class DatasetValidator:  # 数据验证器
    """通用质量验证器"""  # 类说明

    def validate_meta_dir(self, meta_dir: str) -> Dict[str, List[str]]:  # 验证元数据目录
        meta_path = Path(meta_dir)  # 转为路径对象
        issues = {  # 初始化问题列表
            "face_detection_failed": [],  # 人脸检测失败
            "audio_missing": [],  # 音频缺失
            "frame_mismatch": [],  # 帧数不一致
            "small_face": [],  # 人脸过小
        }  # 问题字典
        for meta_file in meta_path.glob("*.json"):  # 遍历 JSON
            try:  # 尝试解析 JSON
                data = json.loads(meta_file.read_text(encoding="utf-8"))  # 读取数据
            except Exception:  # 解析失败
                issues["frame_mismatch"].append(meta_file.name)  # 记录异常
                continue  # 继续下一个
            if not data.get("isvalid", True):  # 判断有效性
                issues["face_detection_failed"].append(meta_file.name)  # 记录问题
            wav_path = data.get("wav_path", "")  # 获取音频路径
            if wav_path and not Path(wav_path).exists():  # 判断音频存在
                issues["audio_missing"].append(meta_file.name)  # 记录问题
            face_size = data.get("face_size", [0, 0])  # 获取人脸尺寸
            if face_size and (face_size[0] < 200 or face_size[1] < 200):  # 判断人脸大小
                issues["small_face"].append(meta_file.name)  # 记录问题
        return issues  # 返回验证结果
