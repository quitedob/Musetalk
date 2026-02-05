"""
Audio processing modules for MuseTalk 2.0

Includes:
- SenseVoice: Fast multilingual audio feature extraction (15x faster than Whisper)
- Whisper compatibility layer
"""

from musetalk.audio.sensevoice_processor import SenseVoiceAudioProcessor

__all__ = ['SenseVoiceAudioProcessor']
