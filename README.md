## constrained-decoder


A decoder that constrains an asr model (in this example Whispers) predictions to match words from a reference text
using word-level timestamps for alignment so as to reduce errors in my any-seguence-length feature extractor 

``` python

import torch
import numpy as np
import whisper
from typing import List, Dict, Optional, Union, Tuple
import re
from difflib import SequenceMatcher


class ConstrainedWhisperDecoder:
    """
    A decoder that constrains an asr model (in this example Whispers) predictions to match words from a reference text
    using word-level timestamps for alignment.
    """
    
    def __init__(
        self,
        whisper_model="tiny",
        device=None,
        temperature=0.0,
        top_p=0.95,
        force_words_threshold=0.8,
    ):
        """
        Initialize a constrained Whisper decoder.
        
        Args:
            whisper_model: Whisper model size to use for timestamp extraction
            device: Device to run on ("cuda", "cpu")
            temperature: Sampling temperature (0 = greedy)
            top_p: Top-p sampling parameter
            force_words_threshold: How aggressively to force words from reference text
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = whisper.load_model(whisper_model, device=self.device)
        self.temperature = temperature
        self.top_p = top_p
        self.force_words_threshold = force_words_threshold
        self.sampling_rate = 16000
        self.dtype = torch.float32
        torch.set_default_dtype(torch.float32)
        torch.set_default_device(device="cuda:0")


    def extract_timestamps(self, audio):
        """Extract word-level timestamps from audio using Whisper"""
        # Ensure audio is in the correct format and dtype
        if isinstance(audio, np.ndarray):
            # Convert to float32 to prevent dtype mismatches
            audio = audio.astype(dtype=np.float32)
        elif isinstance(audio, torch.Tensor):
            # Make sure tensor is float32
            audio = audio.to(dtype=torch.float32)
            # Convert to numpy if needed
            if self.device != "cuda":
                audio = audio.cpu().numpy()
        
        # Normalize audio if needed (follow Whisper's convention)
        if audio.ndim == 2:
            # Convert stereo to mono
            if isinstance(audio, np.ndarray):
                audio = audio.mean(axis=0)
            else:  # PyTorch tensor
                audio = audio.mean(dim=0)
        
        # Ensure audio is within proper range
        if np.abs(audio).max() > 1.0:
            audio = audio / np.abs(audio).max()
        
        # Remove top_p as it's not a valid parameter for DecodingOptions
        result = self.model.transcribe(
            audio=audio, 
            word_timestamps=True,
            temperature=self.temperature
        )
        
        word_timestamps = []
        # Add defensive checking to handle different return formats
        if isinstance(result, dict) and "segments" in result:
            for segment in result["segments"]:
                if isinstance(segment, dict):
                    # Use dictionary access with fallback for "words" key
                    words = segment.get("words", []) if hasattr(segment, "get") else []
                    for word in words:
                        try:
                            word_timestamps.append({
                                "word": word["word"].strip() if isinstance(word["word"], str) else "",
                                "start": word["start"],
                                "end": word["end"],
                                "probability": word.get("probability", 1.0) if hasattr(word, "get") else 1.0
                            })
                        except (KeyError, TypeError, AttributeError) as e:
                            print(f"Error processing word: {word}, error: {str(e)}")
        
        return {
            "text": result.get("text", "") if isinstance(result, dict) else "",
            "words": word_timestamps
        }
        
    def preprocess_reference_text(self, text):
        """Process reference text into words, handling punctuation and case"""
        # Remove punctuation and normalize whitespace
        clean_text = re.sub(r'[^\w\s]', '', text).lower()
        return clean_text.split()
    
    def align_with_reference(
        self, 
        audio, 
        reference_text, 
        max_audio_length=None,
        use_original_timestamps=False
    ):
        """
        Align audio with reference text, optionally truncating both.
        
        Args:
            audio: Audio waveform
            reference_text: Reference text to align with
            max_audio_length: Maximum audio length (in samples) for truncation
            use_original_timestamps: Whether to use original timestamps or adjusted ones
        
        Returns:
            Dict with aligned text, audio, and timestamps
        """
        # Extract timestamps from the full audio
        whisper_result = self.extract_timestamps(audio)
        reference_words = self.preprocess_reference_text(reference_text)
        
        # Create a mapping from Whisper words to reference words using dynamic time warping
        whisper_words = [w["word"].strip().lower() for w in whisper_result["words"]]
        
        # Align using sequence matching
        matcher = SequenceMatcher(None, whisper_words, reference_words)
        
        aligned_words = []
        for op, whisp_start, whisp_end, ref_start, ref_end in matcher.get_opcodes():
            if op == 'equal' or op == 'replace':
                # For matching or replaced segments, map original words to whisper timestamps
                for i, j in zip(range(whisp_start, whisp_end), range(ref_start, ref_end)):
                    if i < len(whisper_words) and j < len(reference_words):
                        word_info = whisper_result["words"][i].copy()
                        # Use original word but whisper timestamp
                        word_info["word"] = reference_words[j]
                        word_info["matched"] = True
                        aligned_words.append(word_info)
            elif op == 'insert':
                # Words in reference but not in whisper - try to estimate position
                if whisp_start > 0 and whisp_start < len(whisper_result["words"]):
                    # Find a reasonable insertion point in the timeline
                    prev_end = whisper_result["words"][whisp_start-1]["end"]
                    next_start = whisper_result["words"][whisp_start]["start"]
                    time_gap = next_start - prev_end
                    
                    for j, ref_word in enumerate(reference_words[ref_start:ref_end]):
                        # Distribute words evenly in the gap
                        ratio = (j + 1) / (ref_end - ref_start + 1)
                        word_time = prev_end + time_gap * ratio
                        
                        aligned_words.append({
                            "word": ref_word,
                            "start": word_time - 0.1,
                            "end": word_time + 0.1,
                            "probability": 0.5,
                            "matched": False
                        })
            elif op == 'delete':
                # Words in whisper but not in reference - skip them
                pass
        
        # Sort by timestamp to ensure correct order
        aligned_words.sort(key=lambda w: w["start"])
        
        # Handle truncation if needed
        if max_audio_length is not None:
            audio_duration = len(audio) / self.sampling_rate
            max_duration = max_audio_length / self.sampling_rate
            
            # Keep only words within the truncated audio
            truncated_words = [w for w in aligned_words if w["start"] < max_duration]
            truncated_text = " ".join(w["word"] for w in truncated_words)
            
            return {
                "audio": audio[:max_audio_length] if max_audio_length < len(audio) else audio,
                "text": truncated_text,
                "timestamps": truncated_words
            }
        
        # Full audio case
        aligned_text = " ".join(w["word"] for w in aligned_words)
        return {
            "audio": audio,
            "text": aligned_text,
            "timestamps": aligned_words
        }
    
    def _word_similarity(self, word1, word2):
        """Calculate similarity between two words (0-1 scale)"""
        # Simple character-level Jaccard similarity
        set1, set2 = set(word1), set(word2)
        if not set1 or not set2:
            return 0.0
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union
        
    def transcribe_with_reference(self, audio, reference_text, max_audio_length=None):
        """
        Transcribe audio with guidance from reference text.
        
        This performs a "constrained decoding" by using the reference text
        to guide Whisper's generation process.
        """
        # First, align the audio with reference text
        alignment_result = self.align_with_reference(
            audio, reference_text, max_audio_length)
        
        # Ensure audio is in float32 format
        if isinstance(audio, np.ndarray):
            audio = audio.astype(np.float32)
        elif isinstance(audio, torch.Tensor):
            audio = audio.to(dtype=torch.float32)
        
        # Run inference with the prompt as guidance
        result = self.model.transcribe(
            audio[:max_audio_length] if max_audio_length else audio,
            initial_prompt=alignment_result["text"],
            condition_on_previous_text=True,
            temperature=self.temperature,
            word_timestamps=True,
        )
        
        return {
            "text": result["text"],
            "segments": result["segments"],
            "alignment": alignment_result
        }

    def batch_process(self, audio_batch, text_batch, max_audio_lengths=None):
        """Process a batch of audio files with their reference texts"""
        results = []
        
        if max_audio_lengths is None:
            max_audio_lengths = [None] * len(audio_batch)
            
        for audio, text, max_len in zip(audio_batch, text_batch, max_audio_lengths):
            result = self.transcribe_with_reference(audio, text, max_len)
            results.append(result)
            
        return results
```
