"""
Engram Viseme Memory Bank

N-gram based memory for explicit phoneme-to-viseme mapping.
Solves "闭嘴" (closed mouth) problem by providing explicit lookup
instead of relying on implicit statistical priors.

Key insights from DeepSeek Engram architecture:
- O(1) deterministic lookup (vs O(L²) attention)
- Deterministic addressing enables prefetching to CPU DRAM
- Separates "memory" (static knowledge) from "compute" (dynamic reasoning)
- Can scale to 100B+ parameters stored in cheap CPU memory

Storage: {audio_n_gram_hash → standard_viseme_latent}
Lookup: O(1) deterministic hash retrieval
Prefetch: CPU→GPU async transfer while processing current layer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
import os
import threading
from collections import OrderedDict


class VisemeMemoryBank:
    """
    N-gram viseme memory bank for explicit phoneme-to-lip mapping.
    
    Key insight from plan2.md:
    - Model's "default closed mouth" is implicit lookup of training distribution
    - Engram makes this explicit: {phoneme_sequence → viseme_latent}
    - O(1) lookup complexity (constant time, regardless of memory size)
    - Deterministic addressing enables CPU prefetching
    
    Hardware-aware design:
    - Memory stored in CPU DRAM (cheap, large capacity)
    - Prefetch to GPU HBM asynchronously
    - Only ~3% latency overhead with proper prefetching
    """
    
    def __init__(self, n_gram: int = 3, num_hashes: int = 8,
                 memory_size: int = 10000, latent_dim: int = 4096,
                 latent_channels: int = 4, latent_size: int = 32,
                 enable_prefetch: bool = True):
        """
        Args:
            n_gram: N-gram order (2=bigram, 3=trigram)
            num_hashes: Number of hash functions for collision reduction
            memory_size: Maximum stored viseme patterns
            latent_dim: Flattened latent dimension
            latent_channels: VAE latent channels (4 for SD, 16 for Flux)
            latent_size: Spatial size of latent (32 for 256px images)
            enable_prefetch: Enable async CPU→GPU prefetching
        """
        self.n_gram = n_gram
        self.num_hashes = num_hashes
        self.memory_size = memory_size
        self.latent_dim = latent_dim
        self.latent_channels = latent_channels
        self.latent_size = latent_size
        self.enable_prefetch = enable_prefetch
        
        # Memory storage (CPU by default for large capacity)
        # Using OrderedDict for LRU-style eviction
        self.memory: OrderedDict[int, torch.Tensor] = OrderedDict()
        self.memory_counts: Dict[int, int] = {}
        
        # Hash projections (random but fixed for deterministic addressing)
        torch.manual_seed(42)
        self.hash_projections = [
            torch.randn(384, 128) * 0.1  # 384 = Whisper feature dim
            for _ in range(num_hashes)
        ]
        
        # Prefetch buffer (GPU-side cache)
        self._prefetch_buffer: Dict[int, torch.Tensor] = {}
        self._prefetch_lock = threading.Lock()
        self._prefetch_thread: Optional[threading.Thread] = None
    
    def _compute_hash(self, audio_features: torch.Tensor, 
                      hash_idx: int) -> int:
        """
        Compute deterministic hash for audio N-gram.
        
        Key property: Hash depends ONLY on input text/audio,
        so we can compute addresses BEFORE model inference.
        This enables prefetching from CPU DRAM to GPU HBM.
        
        Args:
            audio_features: (seq_len, 384) audio features
            hash_idx: Which hash function to use
            
        Returns:
            Integer hash key (deterministic)
        """
        # Extract N-gram (last n_gram features)
        seq_len = audio_features.shape[0]
        if seq_len < self.n_gram:
            # Pad if sequence too short
            pad_len = self.n_gram - seq_len
            audio_features = F.pad(audio_features, (0, 0, pad_len, 0))
        
        n_gram_features = audio_features[-self.n_gram:]  # (n_gram, 384)
        
        # Flatten and project
        flattened = n_gram_features.flatten().cpu()  # (n_gram * 384,)
        
        # Truncate/pad to match projection size
        proj = self.hash_projections[hash_idx]
        proj_input_size = proj.shape[0]
        
        if flattened.shape[0] > proj_input_size:
            flattened = flattened[:proj_input_size]
        elif flattened.shape[0] < proj_input_size:
            flattened = F.pad(flattened, (0, proj_input_size - flattened.shape[0]))
        
        # Deterministic hash via quantized projection
        hash_vec = torch.matmul(flattened, proj)
        
        # Quantize to integer hash (deterministic)
        hash_int = int(torch.sum(torch.sign(hash_vec) * 
                                 torch.arange(1, len(hash_vec) + 1)).item())
        
        return hash_int % (self.memory_size * 10)
    
    def get_prefetch_addresses(self, audio_features: torch.Tensor) -> List[int]:
        """
        Get memory addresses for prefetching (can be called before inference).
        
        This is the key to O(1) lookup with CPU memory:
        1. Compute addresses from audio (deterministic)
        2. Start async prefetch from CPU DRAM to GPU HBM
        3. By the time we need the data, it's already on GPU
        
        Args:
            audio_features: (seq_len, 384) audio features
            
        Returns:
            List of hash keys to prefetch
        """
        addresses = []
        for hash_idx in range(self.num_hashes):
            addr = self._compute_hash(audio_features, hash_idx)
            if addr in self.memory:
                addresses.append(addr)
        return addresses
    
    def prefetch_async(self, addresses: List[int], device: torch.device):
        """
        Asynchronously prefetch memory entries to GPU.
        
        Called before inference to hide CPU→GPU transfer latency.
        """
        if not self.enable_prefetch:
            return
        
        def _prefetch_worker():
            with self._prefetch_lock:
                for addr in addresses:
                    if addr in self.memory and addr not in self._prefetch_buffer:
                        # Transfer to GPU asynchronously
                        self._prefetch_buffer[addr] = self.memory[addr].to(
                            device, non_blocking=True
                        )
        
        self._prefetch_thread = threading.Thread(target=_prefetch_worker)
        self._prefetch_thread.start()
    
    def wait_prefetch(self):
        """Wait for prefetch to complete."""
        if self._prefetch_thread is not None:
            self._prefetch_thread.join()
            self._prefetch_thread = None
    
    def store(self, audio_features: torch.Tensor, 
              viseme_latent: torch.Tensor):
        """
        Store audio-viseme pair in memory.
        
        Args:
            audio_features: (seq_len, 384) audio features
            viseme_latent: (C, H, W) VAE latent of viseme
        """
        latent_flat = viseme_latent.flatten().cpu()
        
        for hash_idx in range(self.num_hashes):
            hash_key = self._compute_hash(audio_features, hash_idx)
            
            if hash_key not in self.memory:
                self.memory[hash_key] = latent_flat.clone()
                self.memory_counts[hash_key] = 1
            else:
                # Exponential moving average update
                count = self.memory_counts[hash_key]
                alpha = 1.0 / (count + 1)
                self.memory[hash_key] = (
                    (1 - alpha) * self.memory[hash_key] + 
                    alpha * latent_flat
                )
                self.memory_counts[hash_key] = count + 1
                # Move to end (LRU)
                self.memory.move_to_end(hash_key)
            
            # LRU eviction when memory full
            while len(self.memory) > self.memory_size:
                # Remove oldest (least recently used)
                oldest_key, _ = self.memory.popitem(last=False)
                if oldest_key in self.memory_counts:
                    del self.memory_counts[oldest_key]
    
    def retrieve(self, audio_features: torch.Tensor,
                 device: torch.device = None) -> Optional[torch.Tensor]:
        """
        Retrieve viseme latent from memory (O(1) lookup).
        
        Args:
            audio_features: (seq_len, 384) audio features
            device: Target device for output
            
        Returns:
            (C, H, W) viseme latent, or None if not found
        """
        # Wait for any pending prefetch
        self.wait_prefetch()
        
        retrieved = []
        
        for hash_idx in range(self.num_hashes):
            hash_key = self._compute_hash(audio_features, hash_idx)
            
            # Check prefetch buffer first (GPU)
            if hash_key in self._prefetch_buffer:
                retrieved.append(self._prefetch_buffer[hash_key])
            elif hash_key in self.memory:
                # Fallback to CPU memory
                tensor = self.memory[hash_key]
                if device is not None:
                    tensor = tensor.to(device)
                retrieved.append(tensor)
        
        # Clear prefetch buffer
        self._prefetch_buffer.clear()
        
        if not retrieved:
            return None
        
        # Average all retrieved latents (collision resolution)
        stacked = torch.stack(retrieved)
        averaged = stacked.mean(dim=0)
        
        # Reshape to latent format
        reshaped = averaged.view(
            self.latent_channels, self.latent_size, self.latent_size
        )
        
        return reshaped
    
    def save(self, path: str):
        """Save memory bank to disk."""
        torch.save({
            'memory': dict(self.memory),  # Convert OrderedDict
            'memory_counts': self.memory_counts,
            'hash_projections': self.hash_projections,
            'config': {
                'n_gram': self.n_gram,
                'num_hashes': self.num_hashes,
                'memory_size': self.memory_size,
                'latent_dim': self.latent_dim,
                'latent_channels': self.latent_channels,
                'latent_size': self.latent_size
            }
        }, path)
        print(f"Memory bank saved: {len(self.memory)} entries -> {path}")
    
    def load(self, path: str) -> bool:
        """Load memory bank from disk."""
        if not os.path.exists(path):
            print(f"Memory bank not found: {path}")
            return False
        
        checkpoint = torch.load(path, weights_only=False)
        self.memory = OrderedDict(checkpoint['memory'])
        self.memory_counts = checkpoint['memory_counts']
        self.hash_projections = checkpoint['hash_projections']
        
        config = checkpoint['config']
        self.n_gram = config['n_gram']
        self.num_hashes = config['num_hashes']
        self.memory_size = config['memory_size']
        self.latent_dim = config['latent_dim']
        self.latent_channels = config['latent_channels']
        self.latent_size = config['latent_size']
        
        print(f"Memory bank loaded: {len(self.memory)} entries <- {path}")
        return True
    
    def build_from_dataset(self, audio_list: List[torch.Tensor],
                           latent_list: List[torch.Tensor],
                           verbose: bool = True):
        """
        Build memory bank from aligned audio-latent pairs.
        
        Args:
            audio_list: List of (seq_len, 384) audio features
            latent_list: List of (C, H, W) viseme latents
            verbose: Print progress
        """
        if verbose:
            print(f"Building Engram memory from {len(audio_list)} samples...")
        
        for i, (audio, latent) in enumerate(zip(audio_list, latent_list)):
            self.store(audio, latent)
            
            if verbose and (i + 1) % 1000 == 0:
                print(f"  Processed {i + 1}/{len(audio_list)} samples")

        if verbose:
            print(f"Memory bank built: {len(self.memory)} unique entries")

    def __len__(self) -> int:
        return len(self.memory)
