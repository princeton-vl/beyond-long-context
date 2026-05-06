"""
M3Agent implementation with proper separation of memory building and inference phases.
Follows VideoLanguageModelInterface for integration with main.py.
"""

import os
import sys
import tempfile
import time
import pickle
import copy
from typing import List, Dict, Any, Optional
import numpy as np
import torch

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.base_interface import VideoLanguageModelInterface
from .gpu_config import GPUConfig
from .memory_builder import MemoryBuilder
from .control_inference import ControlInference
from .video_graph import VideoGraph
from .video_graph_utils import save_video_graph, load_video_graph

class M3Agent(VideoLanguageModelInterface):
    """
    M3-Agent implementation with clean architecture:
    1. Memory Building Phase: Process videos -> Build memory graphs
    2. Control Inference Phase: Load graphs -> Multi-round Q&A
    
    All components must have GPU support or system quits.
    """
    
    def __init__(
        self,
        model_name: str = "ByteDance-Seed/M3-Agent-Control",
        enable_metrics: bool = False,
        max_gpu_mem: Optional[float] = None,
        **kwargs,
    ):
        """Initialize M3-Agent with proper GPU allocation."""
        # Initialize base class first (uses model_id parameter)
        super().__init__(
            model_name,
            enable_metrics=enable_metrics,
            max_gpu_mem=max_gpu_mem,
            **kwargs,
        )
        
        # Storage for current session
        self.video_graphs = {}  # video_id -> VideoGraph object
        self.current_video_id = None
        self.video_counter = 0
        self.main_video_graph = None  # Main VideoGraph for primary video
        self.video_paths = {}  # Store video_id -> video_path mapping
        self.text_entries: List[tuple[str, float]] = []
    
    def _setup_model(self, max_gpu_mem: Optional[float] = None, **kwargs):
        """Model-specific initialization logic."""
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize GPU allocation and components - all must succeed or quit."""
        print(f"Initializing M3-Agent ({self.model_id})...")

        # Create GPU configuration - fails fast if no CUDA
        self.gpu_config = GPUConfig()

        # Allocate GPUs for different components
        self.gpu_config.allocate_vllm(model_size_gb=72)  # Prefers fewer GPUs
        self.gpu_config.allocate_insightface(min_memory_gb=2.0)
        self.gpu_config.allocate_qwen(min_memory_gb=16.0)

        self.gpu_config.print_summary()

        # IMPORTANT: Initialize vLLM FIRST before any other CUDA operations
        # This prevents multiprocessing issues when CUDA context already exists
        self.control_inference = ControlInference(self.gpu_config, enable_metrics=self.enable_metrics)

        # Now that vLLM is initialized, we can safely collect GPU inventory
        self.gpu_config.refresh()

        # Initialize memory builder after vLLM is ready
        self.memory_builder = MemoryBuilder(self.gpu_config, enable_metrics=self.enable_metrics)

        print("M3-Agent initialization completed successfully")

    def _calculate_state_memory_total(self) -> float:
        """Estimate total floats stored across all graphs and text context."""
        total = 0.0

        for graph in self.video_graphs.values():
            if not graph:
                continue

            memory_info = graph.get_memory_size_estimate() or {}
            embedding_stats = memory_info.get('embedding_stats', {})
            content_stats = memory_info.get('content_stats', {})

            # Embedding statistics already report explicit float counts.
            total += float(embedding_stats.get('total_floats', 0))

            # Textual and binary payloads contribute additional stored state.
            total += float(content_stats.get('string_characters', 0))
            total += float(content_stats.get('binary_bytes', 0))

        total += sum(len(text) for text, _ in self.text_entries)
        return total

    def add_video(self, video_frames: np.ndarray, time_start: float = 0.0, 
                  time_end: float = 0.0, video_id: int = 0) -> None:
        """
        Add video to memory graph.
        
        Args:
            video_frames: Video frames as numpy array or base64 list
            time_start: Start time (used for timing context) 
            time_end: End time (used for timing context)
            video_id: Video identifier
        """
        if self.enable_metrics:
            start_time = time.perf_counter()
            # Get cumulative GPU memory across all devices
            baseline_mem = 0
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    baseline_mem += torch.cuda.memory_allocated(i)
        
        print(f"Adding video {video_id} to memory (frames: {len(video_frames) if hasattr(video_frames, '__len__') else 'unknown'})")
        
        # Convert video frames to base64 format for processing - exclude from timing
        base64_frames = self._convert_to_base64(video_frames)
        self.memory_builder.set_frame_limit(len(base64_frames) if base64_frames else None)
        
        # Process video through memory building phase with 30-second chunking
        video_key = f"video_{video_id}"
        
        if video_id == 0:  # Main video - use global clip counter for sequential IDs
            if self.main_video_graph is None:
                self.main_video_graph = VideoGraph()
            
            # Use a global clip counter to ensure sequential clip IDs across chunks
            if not hasattr(self, '_main_video_clip_counter'):
                self._main_video_clip_counter = 0
            
            clip_id = self._main_video_clip_counter
            self._main_video_clip_counter += 1
            
            print(f"Main video chunk: {len(base64_frames)} frames, processing as clip {clip_id}")
            # Process this chunk as a single clip with sequential ID - This is the core model operation
            self.main_video_graph = self.memory_builder.process_video_clip(
                base64_frames, clip_id, self.main_video_graph
            )
            print(f"  ✅ Clip {clip_id} processed successfully")
            
            self.current_video_id = video_key
            
        else:  # Option videos - process as single clip with clip_id=0 (like original)
            clip_id = 0  # Each option video gets clip_id=0 in its own separate graph
            # This is also core model operation
            video_graph = self.memory_builder.process_video_clip(base64_frames, clip_id)
            self.video_graphs[video_key] = video_graph
            print(f"Option video {video_id}: {len(base64_frames)} frames, processing as clip {clip_id}")
        
        # Store main video graph
        if video_id == 0:
            self.video_graphs[video_key] = self.main_video_graph
        
        # Load into control inference for Q&A - exclude from timing as it's file loading
        self.control_inference.load_video_graph(video_key, self.video_graphs[video_key])
        
        print(f"Video {video_id} added to memory graph")
        
        if self.enable_metrics:
            latency = time.perf_counter() - start_time
            # Get peak memory across all GPUs
            peak_mem = 0
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    peak_mem += torch.cuda.max_memory_allocated(i)
                    torch.cuda.reset_peak_memory_stats(i)
            
            peak_mem_increase_mb = max(0, (peak_mem - baseline_mem)) / (1024 * 1024)
            peak_mem_absolute_mb = peak_mem / (1024 * 1024) if torch.cuda.is_available() else 0.0
            
            state_mem = self._calculate_state_memory_total()

            # Collect FLOPS from memory builder
            memory_builder_flops = self.memory_builder.get_and_reset_flops()

            self._record_add_video_metrics(
                latency,
                memory_builder_flops,
                peak_mem_increase_mb,
                peak_mem_absolute_mb,
                time_end,
                state_mem,
            )
    
    def add_text(self, text: str, current_video_time: float = 0.0) -> None:
        """Add text context (used for instructions between videos)."""
        if self.enable_metrics:
            start_time = time.perf_counter()
            baseline_mem = 0
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    baseline_mem += torch.cuda.memory_allocated(i)
        
        # M3-Agent processes this as contextual information
        # In the original, this would be part of the question context
        print(f"Adding text context: {text}")
        # Store text with timestamp for future state calculations
        self.text_entries.append((text, float(current_video_time)))

        if self.enable_metrics:
            latency = time.perf_counter() - start_time
            peak_mem = 0
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    peak_mem += torch.cuda.max_memory_allocated(i)
                    torch.cuda.reset_peak_memory_stats(i)
           
            peak_mem_increase_mb = max(0, (peak_mem - baseline_mem)) / (1024 * 1024)
            peak_mem_absolute_mb = peak_mem / (1024 * 1024) if torch.cuda.is_available() else 0.0
           
            state_mem = self._calculate_state_memory_total()
            self._record_add_text_metrics(
                latency,
                0.0,
                peak_mem_increase_mb,
                peak_mem_absolute_mb,
                current_video_time,
                state_mem,
            )
    
    def ask_question(self, question: str, current_video_time: float = 0.0, max_tokens: int = 1000,
                    max_frames_in_video: Optional[int] = None) -> str:
        """
        Ask question using multi-round inference.
        
        Args:
            question: Question text
            max_tokens: Maximum tokens to generate
            max_frames_in_video: Unused (compatibility)
            
        Returns:
            Generated answer
        """
        question += "(The video you will be searching over is the main video. The text descriptions here are describing the option videos, so use them and their content to search the main video.)"
        if self.enable_metrics:
            start_time = time.perf_counter()
            baseline_mem = 0
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    baseline_mem += torch.cuda.memory_allocated(i)
        
        print(f"Answering question: {question}")
        
        if not self.current_video_id:
            raise RuntimeError("No main video loaded")
        
        # Collect option videos (all videos except main video)
        option_videos = []
        for video_key, video_graph in self.video_graphs.items():
            if video_key != self.current_video_id:
                # Extract option index from video_key (format: "video_1", "video_2", etc.)
                option_index = int(video_key.split('_')[1]) - 1  # Convert video_1 -> option 0, video_2 -> option 1, etc.
                option_videos.append({
                    "video_id": video_key,
                    "option_index": option_index
                })
        
        # Use control inference for multi-round Q&A (match original M3-Agent) - This is the core model operation
        max_rounds = getattr(self.control_inference, "max_control_rounds", 5)
        answer = self.control_inference.answer_question(
            question,
            self.current_video_id,
            option_videos,
            max_rounds,
            max_tokens=max_tokens
        )
        
        print(f"Generated answer: {answer}")
        
        if self.enable_metrics:
            latency = time.perf_counter() - start_time
            peak_mem = 0
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    peak_mem += torch.cuda.max_memory_allocated(i)
                    torch.cuda.reset_peak_memory_stats(i)
            
            peak_mem_increase_mb = max(0, (peak_mem - baseline_mem)) / (1024 * 1024)
            peak_mem_absolute_mb = peak_mem / (1024 * 1024) if torch.cuda.is_available() else 0.0
            
            state_mem = self._calculate_state_memory_total()

            # Collect FLOPS from control inference
            control_inference_flops = self.control_inference.get_and_reset_flops()

            self._record_ask_question_metrics(
                latency,
                control_inference_flops,
                peak_mem_increase_mb,
                peak_mem_absolute_mb,
                current_video_time,
                state_mem,
            )
        print("METRICS", self._metrics)
        return answer

    def get_last_response_token_stats(self) -> Optional[Dict[str, Any]]:
        if not hasattr(self, 'control_inference'):
            return None
        token_stats = self.control_inference.get_last_token_stats()
        if not token_stats:
            return None
        total_output = sum(entry.get('output_tokens', 0) for entry in token_stats)
        total_input = sum(entry.get('input_tokens', 0) for entry in token_stats)
        return {
            'total_output_tokens': total_output,
            'total_input_tokens': total_input,
            'round_token_breakdown': token_stats,
        }
    
    def clear_context(self) -> None:
        """Clear all context for fresh start."""
        print("Clearing M3-Agent context...")
        self.video_graphs.clear()
        self.main_video_graph = None
        self.current_video_id = None
        self.video_counter = 0
        self.text_entries.clear()
        # Reset clip counter for main video
        if hasattr(self, '_main_video_clip_counter'):
            self._main_video_clip_counter = 0
        if hasattr(self, '_reset_state_memory_tracking'):
            self._reset_state_memory_tracking()
        print("Context cleared")

    def _serialize_video_graph(self, video_graph) -> bytes:
        """Serialize a VideoGraph object to bytes."""
        try:
            # Create a deep copy to avoid modifying the original
            graph_copy = copy.deepcopy(video_graph)
            return pickle.dumps(graph_copy)
        except Exception as e:
            print(f"Warning: Failed to serialize VideoGraph: {e}")
            return b''

    def _deserialize_video_graph(self, data: bytes):
        """Deserialize bytes back to a VideoGraph object."""
        try:
            if not data:
                return None
            return pickle.loads(data)
        except Exception as e:
            print(f"Warning: Failed to deserialize VideoGraph: {e}")
            return None

    def save_state(self) -> Dict[str, Any]:
        """
        Save the current model state to memory including full VideoGraph objects.

        Returns a dictionary containing all state that can be restored including
        complete VideoGraph objects with embeddings, memories, and internal state.

        Returns:
            Dict containing:
            - video_graphs_serialized: Serialized VideoGraph objects
            - current_video_id: Currently active video identifier
            - video_counter: Video processing counter
            - main_video_graph_serialized: Serialized main video graph
            - video_paths: Mapping of video IDs to paths
            - main_video_clip_counter: Sequential clip counter for main video
            - control_inference_state: ControlInference internal state
        """
        state = {
            'video_graphs_serialized': {},
            'current_video_id': self.current_video_id,
            'video_counter': self.video_counter,
            'main_video_graph_serialized': None,
            'video_paths': self.video_paths.copy(),
            'main_video_clip_counter': getattr(self, '_main_video_clip_counter', 0),
            'control_inference_state': {},
            'text_entries': copy.deepcopy(self.text_entries),
            '_metrics': None,
        }

        # Serialize all video graphs
        for video_key, video_graph in self.video_graphs.items():
            if video_graph:
                serialized_data = self._serialize_video_graph(video_graph)
                if serialized_data:
                    state['video_graphs_serialized'][video_key] = serialized_data

        # Serialize main video graph separately if different
        if self.main_video_graph and self.main_video_graph not in self.video_graphs.values():
            state['main_video_graph_serialized'] = self._serialize_video_graph(self.main_video_graph)

        # Save ControlInference state (conversation state, last response, etc.)
        if hasattr(self.control_inference, 'last_full_response'):
            state['control_inference_state']['last_full_response'] = self.control_inference.last_full_response

        if self.enable_metrics and self._metrics is not None:
            metrics_dict = {
                'latency_add_video': copy.deepcopy(self._metrics.latency_add_video),
                'latency_add_text': copy.deepcopy(self._metrics.latency_add_text),
                'latency_ask_question': copy.deepcopy(self._metrics.latency_ask_question),
                'flops_add_video': copy.deepcopy(self._metrics.flops_add_video),
                'flops_add_text': copy.deepcopy(self._metrics.flops_add_text),
                'flops_ask_question': copy.deepcopy(self._metrics.flops_ask_question),
                'state_memory_floats': copy.deepcopy(self._metrics.state_memory_floats),
                'state_memory_after_add_video': copy.deepcopy(self._metrics.state_memory_after_add_video),
                'state_memory_after_add_text': copy.deepcopy(self._metrics.state_memory_after_add_text),
                'state_memory_after_ask_question': copy.deepcopy(self._metrics.state_memory_after_ask_question),
                'state_memory_delta_add_video': copy.deepcopy(self._metrics.state_memory_delta_add_video),
                'state_memory_delta_add_text': copy.deepcopy(self._metrics.state_memory_delta_add_text),
                'state_memory_delta_ask_question': copy.deepcopy(self._metrics.state_memory_delta_ask_question),
                'peak_gpu_mem_increase_add_video': copy.deepcopy(self._metrics.peak_gpu_mem_increase_add_video),
                'peak_gpu_mem_increase_add_text': copy.deepcopy(self._metrics.peak_gpu_mem_increase_add_text),
                'peak_gpu_mem_increase_ask_question': copy.deepcopy(self._metrics.peak_gpu_mem_increase_ask_question),
                'peak_gpu_mem_absolute_add_video': copy.deepcopy(self._metrics.peak_gpu_mem_absolute_add_video),
                'peak_gpu_mem_absolute_add_text': copy.deepcopy(self._metrics.peak_gpu_mem_absolute_add_text),
                'peak_gpu_mem_absolute_ask_question': copy.deepcopy(self._metrics.peak_gpu_mem_absolute_ask_question),
                'video_timestamps_add_video': copy.deepcopy(self._metrics.video_timestamps_add_video),
                'video_timestamps_add_text': copy.deepcopy(self._metrics.video_timestamps_add_text),
                'video_timestamps_ask_question': copy.deepcopy(self._metrics.video_timestamps_ask_question),
                'question_correctness_rate': copy.deepcopy(self._metrics.question_correctness_rate),
                'question_dont_know_rate': copy.deepcopy(self._metrics.question_dont_know_rate),
                'question_answered_mask': copy.deepcopy(self._metrics.question_answered_mask),
                'video_timestamps_question_outcome': copy.deepcopy(self._metrics.video_timestamps_question_outcome),
            }
            state['_metrics'] = metrics_dict
        return state

    def load_state(self, state: Dict[str, Any]) -> None:
        """
        Load a previously saved model state including full VideoGraph objects.

        Restores complete state including VideoGraph objects with embeddings,
        memories, and all internal state. This provides true state restoration.

        Args:
            state: State dictionary returned by save_state()
        """
        if not isinstance(state, dict):
            print(f"Warning: Expected dict state, got {type(state)}")
            return

        # Clear current state
        self.clear_context()

        # Restore basic state variables
        self.current_video_id = state.get('current_video_id')
        self.video_counter = state.get('video_counter', 0)
        self.video_paths = state.get('video_paths', {}).copy()
        self.text_entries = copy.deepcopy(state.get('text_entries', []))

        # Restore main video clip counter
        if 'main_video_clip_counter' in state:
            self._main_video_clip_counter = state['main_video_clip_counter']

        # Restore VideoGraph objects from serialized data
        if 'video_graphs_serialized' in state:
            for video_key, serialized_data in state['video_graphs_serialized'].items():
                video_graph = self._deserialize_video_graph(serialized_data)
                if video_graph:
                    self.video_graphs[video_key] = video_graph
                    # Load into control inference for Q&A
                    self.control_inference.load_video_graph(video_key, video_graph)

        # Restore main video graph if separately serialized
        if 'main_video_graph_serialized' in state and state['main_video_graph_serialized']:
            self.main_video_graph = self._deserialize_video_graph(state['main_video_graph_serialized'])

        # Restore ControlInference state
        if 'control_inference_state' in state:
            ci_state = state['control_inference_state']
            if 'last_full_response' in ci_state:
                self.control_inference.last_full_response = ci_state['last_full_response']

        if self.enable_metrics and state.get('_metrics') is not None:
            from models.base_interface import PerformanceMetrics

            if self._metrics is None:
                self._metrics = PerformanceMetrics()

            metrics_data = state['_metrics']
            self._metrics.latency_add_video = metrics_data.get('latency_add_video', []).copy()
            self._metrics.latency_add_text = metrics_data.get('latency_add_text', []).copy()
            self._metrics.latency_ask_question = metrics_data.get('latency_ask_question', []).copy()
            self._metrics.flops_add_video = metrics_data.get('flops_add_video', []).copy()
            self._metrics.flops_add_text = metrics_data.get('flops_add_text', []).copy()
            self._metrics.flops_ask_question = metrics_data.get('flops_ask_question', []).copy()
            self._metrics.state_memory_floats = metrics_data.get('state_memory_floats', []).copy()
            self._metrics.state_memory_after_add_video = metrics_data.get('state_memory_after_add_video', []).copy()
            self._metrics.state_memory_after_add_text = metrics_data.get('state_memory_after_add_text', []).copy()
            self._metrics.state_memory_after_ask_question = metrics_data.get('state_memory_after_ask_question', []).copy()
            self._metrics.state_memory_delta_add_video = metrics_data.get('state_memory_delta_add_video', []).copy()
            self._metrics.state_memory_delta_add_text = metrics_data.get('state_memory_delta_add_text', []).copy()
            self._metrics.state_memory_delta_ask_question = metrics_data.get('state_memory_delta_ask_question', []).copy()
            self._metrics.peak_gpu_mem_increase_add_video = metrics_data.get('peak_gpu_mem_increase_add_video', []).copy()
            self._metrics.peak_gpu_mem_increase_add_text = metrics_data.get('peak_gpu_mem_increase_add_text', []).copy()
            self._metrics.peak_gpu_mem_increase_ask_question = metrics_data.get('peak_gpu_mem_increase_ask_question', []).copy()
            self._metrics.peak_gpu_mem_absolute_add_video = metrics_data.get('peak_gpu_mem_absolute_add_video', []).copy()
            self._metrics.peak_gpu_mem_absolute_add_text = metrics_data.get('peak_gpu_mem_absolute_add_text', []).copy()
            self._metrics.peak_gpu_mem_absolute_ask_question = metrics_data.get('peak_gpu_mem_absolute_ask_question', []).copy()
            self._metrics.video_timestamps_add_video = metrics_data.get('video_timestamps_add_video', []).copy()
            self._metrics.video_timestamps_add_text = metrics_data.get('video_timestamps_add_text', []).copy()
            self._metrics.video_timestamps_ask_question = metrics_data.get('video_timestamps_ask_question', []).copy()
            self._metrics.question_correctness_rate = metrics_data.get('question_correctness_rate', []).copy()
            self._metrics.question_dont_know_rate = metrics_data.get('question_dont_know_rate', []).copy()
            self._metrics.question_answered_mask = metrics_data.get('question_answered_mask', []).copy()
            self._metrics.video_timestamps_question_outcome = metrics_data.get('video_timestamps_question_outcome', []).copy()
            self._sync_state_memory_tracking_from_metrics()

    def _convert_to_base64(self, video_frames) -> List[str]:
        """Convert video frames to base64 format."""
        import cv2
        import base64
        
        base64_frames = []
        
        # Debug: Print input format details
        print(f"  _convert_to_base64 input type: {type(video_frames)}")
        if hasattr(video_frames, 'shape'):
            print(f"  Input shape: {video_frames.shape}")
        if hasattr(video_frames, 'dtype'):
            print(f"  Input dtype: {video_frames.dtype}")
        
        # Handle different input formats
        if isinstance(video_frames, np.ndarray):
            print("  Detected NumPy array")
            # Tensor format - convert each frame
            if video_frames.ndim == 4:  # (frames, height, width, channels)
                for frame in video_frames:
                    # Convert to uint8 if needed
                    if frame.dtype != np.uint8:
                        frame = (frame * 255).astype(np.uint8)
                    
                    # Encode frame to JPEG
                    success, buffer = cv2.imencode('.jpg', frame)
                    if success:
                        base64_str = base64.b64encode(buffer).decode('utf-8')
                        base64_frames.append(base64_str)
                    else:
                        print(f"  WARNING: Failed to encode NumPy frame with shape {frame.shape}")
            else:
                # Single frame
                frame = video_frames
                if frame.dtype != np.uint8:
                    frame = (frame * 255).astype(np.uint8)
                success, buffer = cv2.imencode('.jpg', frame)
                if success:
                    base64_str = base64.b64encode(buffer).decode('utf-8')
                    base64_frames.append(base64_str)
                else:
                    print(f"  WARNING: Failed to encode single NumPy frame with shape {frame.shape}")
        
        # Handle PyTorch tensors
        elif hasattr(video_frames, 'numpy'):  # PyTorch tensor
            print("  Detected PyTorch tensor")
            # Convert PyTorch tensor to numpy array
            if hasattr(video_frames, 'cpu'):
                video_numpy = video_frames.cpu().numpy()
            else:
                video_numpy = video_frames.numpy()
                
            # Process as numpy array
            if video_numpy.ndim == 4:  # Could be (frames, channels, height, width) or (frames, height, width, channels)
                # Check tensor format and convert if needed
                if video_numpy.shape[1] == 3:  # (frames, channels, height, width) - need to transpose
                    print(f"  Converting from (F,C,H,W) to (F,H,W,C) format")
                    video_numpy = video_numpy.transpose(0, 2, 3, 1)  # (frames, height, width, channels)
                
                for frame in video_numpy:
                    # Convert to uint8 if needed
                    if frame.dtype != np.uint8:
                        frame = (frame * 255).astype(np.uint8)
                    
                    # Ensure frame is in HWC format for OpenCV
                    if frame.ndim == 3 and frame.shape[2] == 3:  # (height, width, channels)
                        # Encode frame to JPEG
                        success, buffer = cv2.imencode('.jpg', frame)
                        if success:
                            base64_str = base64.b64encode(buffer).decode('utf-8')
                            base64_frames.append(base64_str)
                        else:
                            print(f"  WARNING: Failed to encode frame with shape {frame.shape}")
                    else:
                        print(f"  WARNING: Unexpected frame shape {frame.shape}, skipping")
            else:
                # Single frame tensor
                frame = video_numpy
                if frame.dtype != np.uint8:
                    frame = (frame * 255).astype(np.uint8)
                success, buffer = cv2.imencode('.jpg', frame)
                if success:
                    base64_str = base64.b64encode(buffer).decode('utf-8')
                    base64_frames.append(base64_str)
                else:
                    print(f"  WARNING: Failed to encode single frame with shape {frame.shape}")
                
        elif isinstance(video_frames, list):
            # Already base64 or frame list
            for frame in video_frames:
                if isinstance(frame, str):
                    # Assume already base64
                    base64_frames.append(frame)
                elif isinstance(frame, np.ndarray):
                    # Convert numpy frame
                    if frame.dtype != np.uint8:
                        frame = (frame * 255).astype(np.uint8)
                    success, buffer = cv2.imencode('.jpg', frame)
                    if success:
                        base64_str = base64.b64encode(buffer).decode('utf-8')
                        base64_frames.append(base64_str)
                    else:
                        print(f"  WARNING: Failed to encode list frame with shape {frame.shape}")
        
        else:
            print(f"  ERROR: Unhandled video_frames type: {type(video_frames)}")
            print(f"  Available attributes: {dir(video_frames)[:10]}...")  # First 10 attributes
        
        print(f"  _convert_to_base64 output: {len(base64_frames)} base64 frames")
        return base64_frames
    
    def get_state(self) -> Dict[str, Any]:
        """Get current state size for monitoring."""
        total_nodes = 0
        total_edges = 0
        total_clips = 0
        
        for video_key, video_graph in self.video_graphs.items():
            stats = video_graph.get_stats()
            total_nodes += stats['total_nodes']
            total_edges += stats['total_edges']
            total_clips += stats['clips_processed']
        
        return {
            'videos_processed': len(self.video_graphs),
            'total_nodes': total_nodes,
            'total_edges': total_edges,
            'total_clips': total_clips,
            'gpu_allocations': len(self.gpu_config.allocations) if self.gpu_config else 0
        }
    
    def get_state_size(self) -> Dict[str, Any]:
        """Alias for compatibility."""
        return self.get_state()
    
    def query_state(self, query: str) -> str:
        """Query current state with natural language."""
        state = self.get_state_size()
        
        if 'memory' in query.lower() or 'node' in query.lower():
            return f"Total nodes: {state['total_nodes']} across {state['videos_processed']} videos"
        elif 'clip' in query.lower():
            return f"Total clips processed: {state['total_clips']}"
        elif 'gpu' in query.lower():
            return f"GPU allocations: {state['gpu_allocations']} components allocated"
        else:
            return f"State: {state}"
