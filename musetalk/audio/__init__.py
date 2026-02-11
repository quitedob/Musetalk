"""MuseTalk 音频模块导出。"""

from musetalk.audio.base_audio_encoder import BaseAudioEncoder, LearnableFeatureAdapter
from musetalk.audio.campplus_encoder import CamPlusAudioEncoder
from musetalk.audio.sensevoice_encoder import SenseVoiceAudioEncoder
from musetalk.audio.sensevoice_processor import SenseVoiceAudioProcessor
from musetalk.audio.whisper_encoder import WhisperAudioEncoder

__all__ = [
    "BaseAudioEncoder",
    "LearnableFeatureAdapter",
    "WhisperAudioEncoder",
    "CamPlusAudioEncoder",
    "SenseVoiceAudioEncoder",
    "SenseVoiceAudioProcessor",
]
