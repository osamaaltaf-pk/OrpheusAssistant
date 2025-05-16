import asyncio
import logging
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Callable, Any
from queue import Queue
import threading
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('pipecat_audio')

@dataclass
class AudioBlock:
    data: np.ndarray
    sample_rate: int
    metadata: dict = None

class AudioProcessor:
    def __init__(self, name: str, process_fn: Callable):
        self.name = name
        self.process_fn = process_fn
        self.input_queue = Queue()
        self.output_queues: List[Queue] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def connect_output(self, queue: Queue):
        self.output_queues.append(queue)

    def process(self, audio_block: AudioBlock) -> AudioBlock:
        try:
            processed = self.process_fn(audio_block)
            return processed
        except Exception as e:
            logger.error(f"Error in processor {self.name}: {e}")
            return audio_block

    def run(self):
        self._running = True
        while self._running:
            try:
                audio_block = self.input_queue.get(timeout=1.0)
                processed_block = self.process(audio_block)
                for queue in self.output_queues:
                    queue.put(processed_block)
            except Exception as e:
                if not isinstance(e, Queue.Empty):
                    logger.error(f"Error in processor {self.name} run loop: {e}")

    def start(self):
        self._thread = threading.Thread(target=self.run)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join()

class AudioPipeline:
    def __init__(self):
        self.processors: List[AudioProcessor] = []
        self._running = False

    def add_processor(self, processor: AudioProcessor):
        if self.processors:
            # Connect to previous processor
            processor.input_queue = self.processors[-1].output_queues[0]
        self.processors.append(processor)

    def start(self):
        self._running = True
        for processor in self.processors:
            processor.start()

    def stop(self):
        self._running = False
        for processor in self.processors:
            processor.stop()

    def process(self, audio_data: np.ndarray, sample_rate: int) -> Optional[AudioBlock]:
        if not self.processors:
            return None

        # Create initial audio block
        audio_block = AudioBlock(data=audio_data, sample_rate=sample_rate)
        
        # Put into first processor's queue
        self.processors[0].input_queue.put(audio_block)
        
        # Get result from last processor
        try:
            result = self.processors[-1].output_queues[0].get(timeout=5.0)
            return result
        except Queue.Empty:
            logger.error("Timeout waiting for pipeline result")
            return None

# Example Audio Processors

class NoiseReducer(AudioProcessor):
    def __init__(self, threshold: float = 0.1):
        super().__init__("noise_reducer", self.reduce_noise)
        self.threshold = threshold

    def reduce_noise(self, audio_block: AudioBlock) -> AudioBlock:
        # Simple noise gate
        audio_data = audio_block.data.copy()
        audio_data[np.abs(audio_data) < self.threshold] = 0
        return AudioBlock(data=audio_data, sample_rate=audio_block.sample_rate)

class AudioNormalizer(AudioProcessor):
    def __init__(self, target_db: float = -20):
        super().__init__("normalizer", self.normalize_audio)
        self.target_db = target_db

    def normalize_audio(self, audio_block: AudioBlock) -> AudioBlock:
        audio_data = audio_block.data.copy()
        if len(audio_data) > 0:
            max_amp = np.max(np.abs(audio_data))
            if max_amp > 0:
                target_amp = 10 ** (self.target_db / 20)
                audio_data = audio_data * (target_amp / max_amp)
        return AudioBlock(data=audio_data, sample_rate=audio_block.sample_rate)

class AudioResampler(AudioProcessor):
    def __init__(self, target_sr: int = 16000):
        super().__init__("resampler", self.resample_audio)
        self.target_sr = target_sr

    def resample_audio(self, audio_block: AudioBlock) -> AudioBlock:
        if audio_block.sample_rate == self.target_sr:
            return audio_block

        # Simple resampling using linear interpolation
        orig_sr = audio_block.sample_rate
        scale = self.target_sr / orig_sr
        n_samples = int(len(audio_block.data) * scale)
        resampled = np.interp(
            np.linspace(0, len(audio_block.data)-1, n_samples),
            np.arange(len(audio_block.data)),
            audio_block.data
        )
        return AudioBlock(data=resampled, sample_rate=self.target_sr)

# GPU-Accelerated Audio Processing
class CUDAAudioProcessor(AudioProcessor):
    def __init__(self, name: str, process_fn: Callable):
        super().__init__(name, process_fn)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"CUDA Audio Processor {name} using device: {self.device}")

    def process(self, audio_block: AudioBlock) -> AudioBlock:
        try:
            # Convert to torch tensor and move to GPU if available
            audio_tensor = torch.from_numpy(audio_block.data).to(self.device)
            processed_tensor = self.process_fn(audio_tensor)
            # Move back to CPU and convert to numpy
            processed_data = processed_tensor.cpu().numpy()
            return AudioBlock(data=processed_data, sample_rate=audio_block.sample_rate)
        except Exception as e:
            logger.error(f"Error in CUDA processor {self.name}: {e}")
            return audio_block

# Example usage
def create_default_pipeline() -> AudioPipeline:
    pipeline = AudioPipeline()
    
    # Add processors
    pipeline.add_processor(NoiseReducer(threshold=0.02))
    pipeline.add_processor(AudioNormalizer(target_db=-20))
    pipeline.add_processor(AudioResampler(target_sr=16000))
    
    return pipeline

if __name__ == "__main__":
    # Test pipeline
    pipeline = create_default_pipeline()
    pipeline.start()
    
    # Create test audio
    test_audio = np.random.randn(16000)  # 1 second of random noise
    result = pipeline.process(test_audio, 16000)
    
    if result:
        logger.info(f"Processed audio shape: {result.data.shape}, sample rate: {result.sample_rate}")
    
    pipeline.stop() 