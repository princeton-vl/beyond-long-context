"""
M3-Agent Memory Building Phase
Processes video clips to build memory graphs for later inference.
"""

import os
import sys
import json
import base64
import torch
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from PIL import Image
from io import BytesIO
from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration, GenerationConfig
from .gpu_config import GPUConfig
from .video_graph import VideoGraph
from .video_graph_utils import save_video_graph
from metrics.flops_calc import (
    m3_agent_memorization_flops,
    qwen3_embedding_0p6b_flops,
)
import librosa
import requests
import cv2
from urllib.parse import urlparse
import re

from .text_embedding import TextEmbeddingBackend, load_api_config
from utils.paths import get_model_cache_dir


CACHE_ROOT = get_model_cache_dir()

def parse_video_caption(video_graph, video_caption):
    """Parse video caption for entity references like <face_1>, <character_2>.

    Args:
        video_graph: VideoGraph instance with character mappings
        video_caption: String containing entity references

    Returns:
        List of (entity_type, entity_id) tuples for valid entities
    """
    def verify_entity(video_graph, entity_str):
        try:
            node_type, node_id = entity_str.split("_")
            node_type = node_type.strip().lower()
            assert node_type in ["face", "voice", "character"]
            node_id = int(node_id)

            # Check character mappings first
            if hasattr(video_graph, 'reverse_character_mappings') and entity_str in video_graph.reverse_character_mappings:
                return (node_type, node_id)
            if hasattr(video_graph, 'character_mappings') and entity_str in video_graph.character_mappings:
                return (node_type, node_id)

            # Check actual nodes
            if ((node_type == 'face' and node_id in video_graph.nodes and video_graph.nodes[node_id].type == 'img') or
                (node_type == 'voice' and node_id in video_graph.nodes and video_graph.nodes[node_id].type == 'voice')):
                return (node_type, node_id)
            return None
        except:
            return None

    pattern = r'<([^<>]*_[^<>]*)>'
    entity_strs = re.findall(pattern, video_caption)
    entities = [verify_entity(video_graph, entity_str) for entity_str in entity_strs]
    return [entity for entity in entities if entity is not None]

class MemoryBuilder:
    """Builds memory graphs from video clips using M3-Agent pipeline."""
    
    def __init__(self, gpu_config: GPUConfig, enable_metrics: bool = False):
        self.gpu_config = gpu_config
        self.enable_metrics = enable_metrics
        self.flops = 0
        self.face_processor = None
        self.voice_processor = None
        self.qwen_processor = None
        self.text_embedder = None
        self.embedding_backend = None

        self._initialize_components()
        self._frame_limit: Optional[int] = None
    
    def _initialize_components(self):
        """Initialize all processing components on assigned GPUs."""
        print("Initializing memory building components...")
        
        # Initialize InsightFace on allocated GPU
        insightface_allocation = self.gpu_config.get_allocation('insightface')
        self._init_insightface(insightface_allocation['gpu'])
        
        # Initialize Qwen VLM on allocated GPU  
        qwen_allocation = self.gpu_config.get_allocation('qwen')
        self._init_qwen_vlm(qwen_allocation['gpu'])
        
        # Initialize text embedder (OpenAI API)
        self._init_text_embedder()
        
        print("✅ All memory building components initialized")
    
    def _init_insightface(self, gpu_id: int):
        """Initialize InsightFace on specific GPU."""
        import torch
        from insightface.model_zoo import model_zoo as model_zoo_impl
        from insightface.app import FaceAnalysis
        import onnxruntime as ort

        # Suppress ONNX warnings
        os.environ['ONNXRUNTIME_LOG_SEVERITY_LEVEL'] = '4'
        ort.set_default_logger_severity(3)
        # Pin ONNX Runtime's per-session thread pools to a deterministic size so
        # it skips the auto-affinity logic that fails on oversubscribed nodes.
        try:
            available_cpus = len(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            available_cpus = os.cpu_count() or 1
        intra_threads = max(1, available_cpus)
        inter_threads = max(1, min(2, available_cpus))
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = intra_threads
        session_options.inter_op_num_threads = inter_threads

        # Force CUDA provider on specific GPU
        providers = [('CUDAExecutionProvider', {'device_id': gpu_id})]
        
        if 'CUDAExecutionProvider' not in ort.get_available_providers():
            print("❌ ONNX Runtime CUDA provider not available")
            print("❌ Install with: pip install onnxruntime-gpu")  
            sys.exit(1)
            
        # Patch insightface's model router once so it forwards session options to
        # onnxruntime. Without this, our custom thread limits would be ignored.
        if not getattr(model_zoo_impl, "_codex_forward_sess_options", False):
            original_get_model = model_zoo_impl.ModelRouter.get_model

            def _forward_sess_options(self, **kwargs):
                if "sess_options" not in kwargs:
                    pending = getattr(model_zoo_impl, "_pending_sess_options", None)
                    if pending is not None:
                        kwargs["sess_options"] = pending
                return original_get_model(self, **kwargs)

            model_zoo_impl.ModelRouter.get_model = _forward_sess_options  # type: ignore[method-assign]
            model_zoo_impl._codex_forward_sess_options = True

        model_zoo_impl._pending_sess_options = session_options
        try:
            self.face_processor = FaceAnalysis(
                name="buffalo_l",
                providers=providers,
            )
        finally:
            model_zoo_impl._pending_sess_options = None
        self.face_processor.prepare(ctx_id=gpu_id)
        print(f"✅ InsightFace initialized on GPU {gpu_id}")
    
    def _init_qwen_vlm(self, gpu_id: int):
        """Initialize Qwen VLM for memory generation with video memory reservation."""
        
        model_name = "ByteDance-Seed/M3-Agent-Memorization"
        
        # Get video memory reservation info
        qwen_allocation = self.gpu_config.get_allocation('qwen')
        video_memory_reserved = qwen_allocation.get('video_memory_reserved_gb', 4.0)
        
        
        # Load model components - exact match to original M3-Agent with memory limit
        max_memory_gb = self.gpu_config.gpu_info[gpu_id]['free_gb'] - video_memory_reserved

        try:
            self.qwen_model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map=None,  # we will explicitly place the model to avoid CPU shards
                attn_implementation="flash_attention_2", 
                cache_dir=CACHE_ROOT,
                max_memory={f"cuda:{gpu_id}": f"{max_memory_gb:.1f}GiB"}
            )
            self.qwen_model.to(f"cuda:{gpu_id}")
        except Exception as e:
            print(f"❌ Qwen M3-Agent-Memorization: Failed to load on GPU {gpu_id} (reserved {video_memory_reserved}GB for video)")
            print(f"   Error: {str(e)}")
            raise
        self.qwen_model.eval()
        
        self.qwen_processor = Qwen2_5OmniProcessor.from_pretrained(
            model_name,
            cache_dir=CACHE_ROOT
        )

        print(f"✅ Qwen VLM initialized on GPU {gpu_id}")
    
    def _init_text_embedder(self):
        """Initialize text embedder used for memory summarization."""
        config_path = os.path.join(os.path.dirname(__file__), "configs", "processing_config.json")
        api_config_path = os.path.join(os.path.dirname(__file__), "configs", "api_config.json")
        embedding_model = "text-embedding-3-large"
        embedding_device = "auto"
        if os.path.exists(config_path):
            try:
                processing_config = json.load(open(config_path))
                embedding_model = processing_config.get("embedding_model", embedding_model)
                embedding_device = processing_config.get("embedding_device", embedding_device)
            except Exception as exc:
                print(f"Warning: failed to load embedding config: {exc}")

        if embedding_device == "cuda":
            try:
                qwen_allocation = self.gpu_config.get_allocation('qwen')
                embedding_device = f"cuda:{qwen_allocation['gpu']}"
            except Exception:
                embedding_device = "cuda"

        api_config = load_api_config(api_config_path)
        self.embedding_device = embedding_device
        self.embedding_backend = TextEmbeddingBackend(
            embedding_model,
            api_config=api_config,
            target_dim=1536,
            device=embedding_device,
        )
        self.text_embedder = embedding_model
        print(f"✅ Text embedder configured: {self.text_embedder} ({self.embedding_backend.provider})")
    
    def process_video_clip(self, base64_frames: List[str], clip_id: int, video_graph: VideoGraph = None) -> VideoGraph:
        """Process single video clip and build/update VideoGraph."""
        print(f"Processing clip {clip_id} with {len(base64_frames)} frames...")
        
        # Create new VideoGraph if not provided
        if video_graph is None:
            video_graph = VideoGraph()
        
        # Step 1: Face detection and clustering
        faces = self._detect_and_cluster_faces(base64_frames)
        print(f"  Found {len(faces)} faces")
        
        # Step 2: Add face nodes to VideoGraph
        self._update_video_graph_with_faces(video_graph, faces)
        
        # Step 3: Generate memories using Qwen VLM
        episodic_memories, semantic_memories = self._generate_memories(
            base64_frames, faces, clip_id
        )
        print(f"  Generated {len(episodic_memories)} episodic, {len(semantic_memories)} semantic memories")
        
        # Step 4: Create memory embeddings and add to VideoGraph
        self._add_memories_to_graph(video_graph, episodic_memories, semantic_memories, clip_id)
        print(f"  Added memories to VideoGraph")

        return video_graph

    def set_frame_limit(self, limit: Optional[int]) -> None:
        """Set the maximum number of frames to sample from raw video files."""

        if limit is not None and limit <= 0:
            limit = None
        self._frame_limit = limit
    
    def _detect_and_cluster_faces(self, base64_frames: List[str]) -> List[Dict]:
        """Detect and cluster faces using InsightFace."""
        import cv2
        
        faces = []
        for frame_idx, frame_b64 in enumerate(base64_frames):
            # Decode frame
            img_bytes = base64.b64decode(frame_b64)
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if img is None:
                continue
                
            # Detect faces
            detected_faces = self.face_processor.get(img)
            
            for face in detected_faces:
                bbox = [int(x) for x in face.bbox.astype(int).tolist()]
                embedding = [float(x) for x in face.normed_embedding.tolist()]
                
                # Extract face crop
                face_img = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                _, buffer = cv2.imencode('.jpg', face_img)
                face_base64 = base64.b64encode(buffer).decode('utf-8')
                
                face_info = {
                    "frame_id": frame_idx,
                    "bounding_box": bbox,
                    "face_emb": embedding,
                    "cluster_id": -1,  # Will be set by clustering
                    "extra_data": {
                        "face_type": "ortho",  # Simplified
                        "face_base64": face_base64,
                        "face_detection_score": str(face.det_score),
                        "face_quality_score": str(np.linalg.norm(face.embedding))
                    }
                }
                faces.append(face_info)
        
        # Cluster faces using HDBSCAN (exact match to m3-agent-fixed)
        faces = self._cluster_faces_hdbscan(faces)
        return faces
    
    def _cluster_faces_hdbscan(self, faces: List[Dict]) -> List[Dict]:
        """Port EXACT HDBSCAN clustering from m3-agent-fixed."""
        import hdbscan
        import numpy as np

        if len(faces) < 2:
            for i, face in enumerate(faces):
                face['cluster_id'] = i
            return faces

        face_embeddings = [face["face_emb"] for face in faces]
        face_detection_scores = [float(face["extra_data"]["face_detection_score"]) for face in faces]
        face_quality_scores = [float(face["extra_data"]["face_quality_score"]) for face in faces]

        # EXACT thresholds from m3-agent-fixed
        detection_threshold = 0.8
        quality_threshold = 20
        good_mask = [(face_detection_scores[i] >= detection_threshold and
                      face_quality_scores[i] >= quality_threshold) for i in range(len(faces))]

        face_embeddings = np.array(face_embeddings)
        good_embeddings = face_embeddings[good_mask]

        all_labels = [-1] * len(faces)

        if len(good_embeddings) >= 2:
            # EXACT distance calculation from m3-agent-fixed
            good_similarity = np.dot(good_embeddings, good_embeddings.T)
            good_distances = 1 - good_similarity
            good_distances = np.maximum(good_distances, 0).astype(np.float64)

            # EXACT HDBSCAN parameters
            good_clusterer = hdbscan.HDBSCAN(min_cluster_size=2, metric="precomputed")
            good_labels = good_clusterer.fit_predict(good_distances)

            # Assign labels to good faces
            good_idx = 0
            for i, is_good in enumerate(good_mask):
                if is_good:
                    all_labels[i] = good_labels[good_idx]
                    good_idx += 1

        result_faces = []
        for i, face in enumerate(faces):
            face_copy = face.copy()
            face_copy["cluster_id"] = all_labels[i]
            result_faces.append(face_copy)

        return result_faces
    
    def _generate_memories(self, base64_frames: List[str], faces: List[Dict], clip_id: int) -> Tuple[List[str], List[str]]:
        """Generate episodic and semantic memories using Qwen VLM - exact match to original M3-Agent."""
        print(f"Generate memories for clip {clip_id}: {len(base64_frames)} frames, {len(faces)} faces")
        
        # Create face features dict with base64 images
        face_features = {}
        for i, face in enumerate(faces):
            if face['cluster_id'] != -1:
                face_key = f"<face_{face['cluster_id']}>"
                if face_key not in face_features:
                    face_features[face_key] = face['extra_data']['face_base64']
        
        # Create temporary video file from frames for VLM processing
        import tempfile
        import os
        temp_video_path = self._create_temp_video_from_frames(base64_frames)
        
        print(f"Created temp video: {temp_video_path}")
        
        # Prepare input for Qwen VLM - exact match to original structure
        input_content = []
        
        # Add the M3-Agent memory generation prompt
        prompt = """You will be given a video and a set of character features. Each feature is either a face (represented by a video frame with a bounding box) or a voice (represented by one or more speech segments, each with MM:SS start and end times, and transcript content). Each feature has a unique ID enclosed in angle brackets. Some features may belong to the same character.
Your task consists of two parts:
1. Video Description:
Generate a detailed and cohesive description of the current video clip. Use the provided feature IDs as references to characters (when applicable). Your description should cover all observable and inferable events. Each description should focus on a single atomic event or fact.
2. High-Level Conclusions:
Generate high-level reasoning-based conclusions that go beyond surface-level observations. Use logical inference to identify character intentions, relationships, and identities. If a face and a voice feature refer to the same character, indicate it using this exact format: Equivalence: <face_x>, <voice_y>
Output Format:
Your output must be a JSON object with the following structure:
{
    "video_descriptions": [
        "...",  // each string is one atomic event description
        "..."
    ],
    "high_level_conclusions": [
        "...",  // each string is one high-level inference or identity resolution
        "Equivalence: <face_1>, <voice_2>"
    ]
}
Please only return the valid JSON object, without any additional explanation or formatting."""
        
        input_content.append({"type": "text", "content": prompt})
        
        # Use temp video file path instead of base64 data for qwen_omni_utils
        input_content.append({
            "type": "video_url", 
            "content": temp_video_path,
        })
        
        input_content.append({
            "type": "text",
            "content": "Face features:"
        })
        
        # Add face images with labels - exact match to original structure
        face_images_list = []
        for face_id, face_b64 in face_features.items():
            face_images_list.append((face_id + ":", face_b64))
        
        input_content.append({
            "type": "images/jpeg",
            "content": face_images_list,
        })
        
        input_content.append({
            "type": "text",
            "content": "Voice features:"
        })
        
        input_content.append({
            "type": "text",
            "content": "{}",  # Empty voice features for now
        })
        
        # Parse JSON response using EXACT m3-agent-fixed approach with retries
        import json
        import os

        # Load processing config for MAX_RETRIES
        config_path = os.path.join(os.path.dirname(__file__), "configs", "processing_config.json")
        processing_config = json.load(open(config_path))
        MAX_RETRIES = processing_config.get("max_memorization_retries", processing_config.get("max_retries", 20))

        # Use the exact approach from m3-agent-fixed: validate_and_fix_json
        def refine_json_str(invalid_json):
            """Clean and format JSON string by removing markdown code blocks."""
            fixed_json = invalid_json.strip("```json").strip("```python").strip("```").strip()
            return fixed_json

        def validate_and_fix_json(invalid_json):
            fixed_json = refine_json_str(invalid_json)
            try:
                return json.loads(fixed_json)
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
                return None

        # EXACT m3-agent-fixed logic with retry loop
        messages = self._generate_messages_original(input_content)
        epi_key = "video_descriptions"
        sem_key = "high_level_conclusions"

        memory_data = None
        for i in range(MAX_RETRIES):
            response, prompt_tokens, generated_tokens = self._get_response_original(messages)
            print(f"VLM response (attempt {i+1}): {response[:200]}..." if len(response) > 200 else f"VLM response (attempt {i+1}): {response}")

            # Calculate FLOPS if metrics enabled
            if self.enable_metrics and i == 0:  # Only count FLOPS on first attempt to avoid double counting
                # Get video frame dimensions from first frame
                if base64_frames:
                    img_bytes = base64.b64decode(base64_frames[0])
                    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                    import cv2
                    first_frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    if first_frame is not None:
                        height, width, _ = first_frame.shape
                        vision_frames = len(base64_frames)

                        output_tokens = generated_tokens

                        # Calculate FLOPS using m3_agent_memorization_flops
                        memorization_flops = m3_agent_memorization_flops(
                            vision_frames=vision_frames,
                            vision_height=height,
                            vision_width=width,
                            lang_prompt_len=prompt_tokens,
                            num_generated=output_tokens,
                            do_backward=False
                        )
                        # Extract total FLOPS from the result dict
                        total_flops = memorization_flops.get('total_flops', 0)
                        self.flops += total_flops

            if not response:
                response = "[]"
            memory_data = validate_and_fix_json(response)
            if memory_data is not None:
                break

        if memory_data is None:
            memory_data = {
                epi_key: [],
                sem_key: []
            }

        print(f"Parsed memory data: {str(memory_data)[:200]}...")

        # Clean up temp video file
        if os.path.exists(temp_video_path):
            os.unlink(temp_video_path)
            print(f"Cleaned up temp video: {temp_video_path}")
        
        # Handle both original format (singular) and our format (plural)
        episodic_memories = memory_data.get("video_description", memory_data.get("video_descriptions", []))
        semantic_memories = memory_data.get("high_level_conclusions", [])
        
        if not episodic_memories and not semantic_memories:
            print(f"ERROR: VLM generated no memories: {memory_data}")
            raise RuntimeError(f"Qwen VLM generated no memories for clip {clip_id}")
        
        print(f"Parsed {len(episodic_memories)} episodic, {len(semantic_memories)} semantic memories")
        
        return episodic_memories, semantic_memories
    
    def _create_temp_video_from_frames(self, base64_frames: List[str]) -> str:
        """Create temporary video file from base64 frames for VLM processing."""
        import tempfile
        import cv2
        import base64
        import numpy as np
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
            temp_path = temp_file.name
        
        if not base64_frames:
            raise RuntimeError("No frames provided for video creation")
        
        # Decode first frame to get dimensions
        img_bytes = base64.b64decode(base64_frames[0])
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        first_frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if first_frame is None:
            raise RuntimeError("Failed to decode first frame")
        
        height, width, _ = first_frame.shape
        fps = 5  # Match original M3-Agent fps
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))
        
        print(f"Creating temp video: {len(base64_frames)} frames at {fps}fps, {width}x{height}")
        
        try:
            for frame_b64 in base64_frames:
                # Decode frame
                img_bytes = base64.b64decode(frame_b64)
                img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                
                if frame is not None:
                    writer.write(frame)
                else:
                    print("Warning: Failed to decode frame, skipping")
        finally:
            writer.release()
        
        print(f"Temp video created: {temp_path}")
        return temp_path
    
    def _generate_messages_original(self, inputs):
        """Original M3-Agent generate_messages function copied exactly."""
        messages = []
        content = []
        for input in inputs:
            if not input["content"]:
                continue
            if input["type"] == "text":
                content.append({"type": "text", "text": input["content"]})
            elif input["type"] in ["images/jpeg", "images/png"]:
                img_format = input["type"].split("/")[1]
                if isinstance(input["content"][0], str):
                    content.extend(
                        [
                            {
                                "type": "image",
                                "image": f"data:image;base64,{img}",
                            }
                            for img in input["content"]
                        ]
                    )
                else:
                    for img in input["content"]:
                        content.append({
                            "type": "text",
                            "text": img[0],
                        })
                        content.append({
                            "type": "image",
                            "image": f"data:image;base64,{img[1]}"
                        })
            elif input["type"] in ["video_url", "video_base64/mp4", "video_base64/webm"]:
                content.append(
                    {
                        "type": "video",
                        "video": input["content"],
                    }
                )
            else:
                raise ValueError(f"Invalid input type: {input['type']}")
        messages.append({"role": "user", "content": content})
        return messages
    
    def _get_response_original(self, messages):
        """Original M3-Agent get_response function copied exactly."""
        import json
        from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration, GenerationConfig
        import torch
        
        # Load processing config
        config_path = os.path.join(os.path.dirname(__file__), "configs", "processing_config.json")
        processing_config = json.load(open(config_path))
        temp = processing_config["temperature"]
        
        # Use our already initialized models
        thinker = self.qwen_model
        processor = self.qwen_processor
        
        text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        generation_config = GenerationConfig(pad_token_id=151643, bos_token_id=151644, eos_token_id=151645)
        
        USE_AUDIO_IN_VIDEO = False
        # Import and use the real qwen_omni_utils function
        try:
            from qwen_omni_utils import process_mm_info
            audios, images, videos = process_mm_info(messages, use_audio_in_video=USE_AUDIO_IN_VIDEO)
        except ImportError as e:
            print(f"Failed to import qwen_omni_utils: {e}")
            audios, images, videos = self._process_mm_info_original(messages, use_audio_in_video=USE_AUDIO_IN_VIDEO)
        inputs = processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=USE_AUDIO_IN_VIDEO,
        )
        inputs = inputs.to(device=thinker.device)
        inputs = inputs.to(thinker.dtype)
        prompt_token_count = int(inputs.input_ids.size(1))

        # Inference: Generation of the output text and audio
        with torch.no_grad():
            generation = thinker.generate(
                **inputs,
                generation_config=generation_config,
                use_audio_in_video=USE_AUDIO_IN_VIDEO,
                max_new_tokens=8192,
                do_sample=True,
                temperature=temp,
            )
            generate_ids = generation[:, prompt_token_count:]
            response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            generated_token_count = int(generate_ids.size(1))
            
        # Clean up
        del generation
        del generate_ids
        del inputs
        torch.cuda.empty_cache()

        return response, prompt_token_count, generated_token_count
    
    def _process_mm_info_original(self, messages, use_audio_in_video=False):
        """Original M3-Agent process_mm_info function copied exactly."""
        audios = []
        images = []
        videos = []
        
        for message in messages:
            if isinstance(message["content"], list):
                for content in message["content"]:
                    if content["type"] == "image":
                        images.append(content["image"])
                    elif content["type"] == "video":
                        videos.append(content["video"])
        
        return audios, images, videos
    
    def _create_memory_embeddings(self, memories: List[str]) -> List[List[float]]:
        """Create embeddings for memories using configured backend."""
        if not memories:
            return []

        if not self.embedding_backend:
            return [[0.0] * 1536 for _ in memories]

        result = self.embedding_backend.embed_texts(memories)
        embeddings = result.vectors

        if len(embeddings) < len(memories):
            missing = len(memories) - len(embeddings)
            embeddings.extend([[0.0] * self.embedding_backend.target_dim for _ in range(missing)])

        if self.enable_metrics and memories:
            provider = result.provider.lower()
            if provider != "qwen":
                raise RuntimeError(
                    "Performance metrics for M3-Agent require the Qwen embedding backend; "
                    "configure the text embedding model accordingly."
                )
            embed_flops = 0.0
            token_lengths = self.embedding_backend.token_lengths(memories, role="document")
            for seq_len in token_lengths:
                if seq_len > 0:
                    embed_flops += qwen3_embedding_0p6b_flops(1, seq_len)
            self.flops += embed_flops

        return embeddings

    def get_and_reset_flops(self) -> float:
        """Get accumulated FLOPS and reset counter."""
        flops = self.flops
        self.flops = 0
        return flops

    def _update_video_graph_with_faces(self, video_graph: VideoGraph, faces: List[Dict]):
        """Add face nodes to VideoGraph following M3-Agent pattern."""
        # Group faces by cluster_id
        face_clusters = {}
        for face in faces:
            cluster_id = face['cluster_id']
            if cluster_id == -1:
                continue  # Skip unmatched faces
                
            if cluster_id not in face_clusters:
                face_clusters[cluster_id] = []
            face_clusters[cluster_id].append(face)
        
        # Add face nodes to VideoGraph
        for cluster_id, cluster_faces in face_clusters.items():
            face_info = {
                "embeddings": [face["face_emb"] for face in cluster_faces],
                "contents": [face["extra_data"]["face_base64"] for face in cluster_faces],
            }
            
            # Search for existing face nodes
            matched_nodes = video_graph.search_img_nodes(face_info)
            if len(matched_nodes) > 0:
                # Update existing node
                matched_node = matched_nodes[0][0]
                video_graph.update_node(matched_node, face_info)
                for face in cluster_faces:
                    face["matched_node"] = matched_node
            else:
                # Create new node
                matched_node = video_graph.add_img_node(face_info)
                for face in cluster_faces:
                    face["matched_node"] = matched_node
    
    def _add_memories_to_graph(self, video_graph: VideoGraph, episodic_memories: List[str],
                              semantic_memories: List[str], clip_id: int):
        """Add episodic and semantic memories with complex semantic processing logic.

        Port of exact semantic memory processing from m3-agent-fixed.
        """
        def insert_memory(video_graph, memory_data, clip_id, memory_type):
            new_node_id = video_graph.add_text_node(memory_data, clip_id, memory_type)
            entities = parse_video_caption(video_graph, memory_data['contents'][0])
            for entity_type, entity_id in entities:
                if entity_id in video_graph.nodes:
                    video_graph.add_edge(new_node_id, entity_id)
            return new_node_id

        all_memories = episodic_memories + semantic_memories
        if not all_memories:
            return
        embeddings = self._create_memory_embeddings(all_memories)

        # Add episodic memories (simple append)
        for i, memory in enumerate(episodic_memories):
            memory_data = {
                'contents': [memory],
                'embeddings': [embeddings[i]] if i < len(embeddings) else []
            }
            insert_memory(video_graph, memory_data, clip_id, 'episodic')

        # Add semantic memories (complex similarity logic - EXACT from m3-agent-fixed)
        semantic_start_idx = len(episodic_memories)
        for i, memory in enumerate(semantic_memories):
            embedding_idx = semantic_start_idx + i
            memory_data = {
                'contents': [memory],
                'embeddings': [embeddings[embedding_idx]] if embedding_idx < len(embeddings) else []
            }

            entities = parse_video_caption(video_graph, memory)

            if len(entities) == 0:
                insert_memory(video_graph, memory_data, clip_id, 'semantic')
                continue

            # EXACT logic from memory_processing_qwen.py lines 212-243
            positive_threshold = 0.85
            negative_threshold = 0

            first_entity_id = entities[0][1]
            if first_entity_id in video_graph.nodes:
                related_nodes = video_graph.get_connected_nodes(first_entity_id, type=['semantic'])
            else:
                related_nodes = []

            create_new_node = True
            memory_embedding = embeddings[embedding_idx] if embedding_idx < len(embeddings) else None

            for existing_node_id in related_nodes:
                if existing_node_id not in video_graph.nodes:
                    continue
                existing_node = video_graph.nodes[existing_node_id]
                if 'contents' not in existing_node.metadata or not existing_node.metadata['contents']:
                    continue

                existing_content = existing_node.metadata['contents'][0]
                existing_entities = parse_video_caption(video_graph, existing_content)

                # Check if entities are subset of existing node entities
                if all(entity in existing_entities for entity in entities):
                    if (memory_embedding and existing_node.embeddings and
                        len(memory_embedding) > 0 and len(existing_node.embeddings[0]) > 0):

                        import numpy as np
                        mem_emb = np.array(memory_embedding)
                        exist_emb = np.array(existing_node.embeddings[0])
                        similarity = np.dot(mem_emb, exist_emb) / (np.linalg.norm(mem_emb) * np.linalg.norm(exist_emb))

                        if similarity > positive_threshold:
                            video_graph.reinforce_node(existing_node_id)
                            create_new_node = False
                            break
                        elif similarity < negative_threshold:
                            video_graph.weaken_node(existing_node_id)
                            create_new_node = False
                            break

            if create_new_node:
                insert_memory(video_graph, memory_data, clip_id, 'semantic')

    def _process_mm_info(self, messages: List[Dict[str, Any]], use_audio_in_video: bool = False) -> Tuple[Any, List[Any], List[Any]]:
        """
        Process multimedia information from messages for Qwen2_5OmniThinkerForConditionalGeneration.
        Clean implementation without mmagent dependency.
        
        Args:
            messages: List of message dictionaries containing multimedia content
            use_audio_in_video: Whether to extract audio from video files
            
        Returns:
            Tuple of (audios, images, videos) processed for Qwen model
        """
        audios = None
        images = []
        videos = []
        
        # Process all messages for multimedia content
        for message in messages:
            content = message.get("content", [])
            if isinstance(content, str):
                # For simple text messages, check if it contains video frames
                continue
                
            # Handle list of content items
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        item_type = item.get("type", "")
                        
                        if item_type == "image_url":
                            image = self._process_image_item(item)
                            if image is not None:
                                images.append(image)
                                
                        elif item_type == "video":
                            video_frames = self._process_video_item(item, use_audio_in_video)
                            if video_frames is not None:
                                videos.append(video_frames)
                                
                        elif item_type == "audio" and use_audio_in_video:
                            audio = self._process_audio_item(item)
                            if audio is not None:
                                audios = audio
        
        # Return None instead of empty lists if no content found (Qwen processor expects None)
        if not images:
            images = None
        if not videos:
            videos = None
            
        return audios, images, videos

    def _process_image_item(self, item: Dict[str, Any]) -> Image.Image:
        """Process an image item from message content."""
        try:
            url_data = item.get("image_url", {})
            url = url_data.get("url", "")
            
            if url.startswith("data:image"):
                # Base64 encoded image
                header, data = url.split(",", 1)
                image_data = base64.b64decode(data)
                image = Image.open(BytesIO(image_data))
                if image.mode == "RGBA":
                    # Convert RGBA to RGB with white background
                    rgb_image = Image.new("RGB", image.size, (255, 255, 255))
                    rgb_image.paste(image, mask=image.split()[-1])
                    image = rgb_image
                return self._smart_resize_image(image)
                
            elif url.startswith(("http://", "https://")):
                # URL image
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content))
                if image.mode == "RGBA":
                    rgb_image = Image.new("RGB", image.size, (255, 255, 255))
                    rgb_image.paste(image, mask=image.split()[-1])
                    image = rgb_image
                return self._smart_resize_image(image)
                
            elif url.startswith("file://") or os.path.isfile(url):
                # Local file
                path = url.replace("file://", "")
                image = Image.open(path)
                if image.mode == "RGBA":
                    rgb_image = Image.new("RGB", image.size, (255, 255, 255))
                    rgb_image.paste(image, mask=image.split()[-1])
                    image = rgb_image
                return self._smart_resize_image(image)
                
        except Exception as e:
            return None
            
    def _process_video_item(self, item: Dict[str, Any], use_audio_in_video: bool) -> List[Image.Image]:
        """Process a video item from message content."""
        try:
            url = item.get("video_url", item.get("url", ""))
            
            if os.path.isfile(url) or url.startswith("file://"):
                path = url.replace("file://", "")
                return self._extract_video_frames(path)
                
        except Exception as e:
            return None
            
    def _process_audio_item(self, item: Dict[str, Any]) -> np.ndarray:
        """Process an audio item from message content."""
        try:
            url = item.get("audio_url", item.get("url", ""))
            
            if url.startswith("data:audio"):
                # Base64 encoded audio
                header, data = url.split(",", 1)
                audio_data = base64.b64decode(data)
                # Save temporarily and load with librosa
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    f.write(audio_data)
                    temp_path = f.name
                audio, sr = librosa.load(temp_path, sr=16000, mono=True)
                os.unlink(temp_path)
                return audio
                
            elif os.path.isfile(url) or url.startswith("file://"):
                path = url.replace("file://", "")
                audio, sr = librosa.load(path, sr=16000, mono=True)
                return audio
                
        except Exception as e:
            return None

    def _smart_resize_image(self, image: Image.Image) -> Image.Image:
        """
        Smart resize following M3-Agent patterns.
        Ensures dimensions divisible by 28 and within pixel limits.
        """
        IMAGE_FACTOR = 28
        MIN_PIXELS = 3136
        MAX_PIXELS = 12845056
        MAX_RATIO = 200
        
        width, height = image.size
        current_pixels = width * height
        
        # Check if resize needed
        if (width % IMAGE_FACTOR == 0 and height % IMAGE_FACTOR == 0 and 
            MIN_PIXELS <= current_pixels <= MAX_PIXELS and
            max(width, height) / min(width, height) <= MAX_RATIO):
            return image
            
        # Calculate target dimensions
        aspect_ratio = width / height
        
        # Constrain aspect ratio
        if aspect_ratio > MAX_RATIO:
            aspect_ratio = MAX_RATIO
        elif aspect_ratio < 1/MAX_RATIO:
            aspect_ratio = 1/MAX_RATIO
            
        # Find optimal dimensions within pixel constraints
        target_pixels = min(MAX_PIXELS, max(MIN_PIXELS, current_pixels))
        
        # Calculate new dimensions
        new_height = int((target_pixels / aspect_ratio) ** 0.5)
        new_width = int(new_height * aspect_ratio)
        
        # Round to IMAGE_FACTOR
        new_width = ((new_width + IMAGE_FACTOR - 1) // IMAGE_FACTOR) * IMAGE_FACTOR
        new_height = ((new_height + IMAGE_FACTOR - 1) // IMAGE_FACTOR) * IMAGE_FACTOR
        
        # Final check
        if new_width * new_height > MAX_PIXELS:
            scale = (MAX_PIXELS / (new_width * new_height)) ** 0.5
            new_width = int(new_width * scale / IMAGE_FACTOR) * IMAGE_FACTOR
            new_height = int(new_height * scale / IMAGE_FACTOR) * IMAGE_FACTOR
            
        return image.resize((new_width, new_height), Image.LANCZOS)

    def _extract_video_frames(self, video_path: str) -> List[Image.Image]:
        """Extract frames from video following M3-Agent patterns."""
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Smart frame sampling - match M3-Agent logic
            target_fps = 2.0
            min_frames = 4
            max_frames = self._frame_limit if self._frame_limit is not None else 768

            if fps > 0:
                frame_interval = max(1, int(fps / target_fps))
                target_frame_count = min(max_frames, max(min_frames, total_frames // frame_interval))
            else:
                target_frame_count = min_frames
                frame_interval = max(1, total_frames // target_frame_count)
            
            frames = []
            frame_indices = np.linspace(0, total_frames - 1, target_frame_count, dtype=int)
            
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_frame = Image.fromarray(frame_rgb)
                    resized_frame = self._smart_resize_image(pil_frame)
                    frames.append(resized_frame)
                    
            cap.release()
            return frames if frames else None
            
        except Exception as e:
            return None


def test_memory_builder():
    """Test memory builder with minimal data."""
    print("Testing Memory Builder...")
    
    # Create GPU config
    gpu_config = GPUConfig()
    
    # Allocate components
    gpu_config.allocate_vllm(model_size_gb=12, min_gpus=2) 
    gpu_config.allocate_insightface(min_memory_gb=2.0)
    gpu_config.allocate_qwen(min_memory_gb=8.0)
    gpu_config.print_summary()
    
    # Create memory builder
    builder = MemoryBuilder(gpu_config)
    
    # Create test frames
    import cv2
    test_frames = []
    for i in range(3):
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        _, buffer = cv2.imencode('.jpg', img)
        base64_str = base64.b64encode(buffer).decode('utf-8')
        test_frames.append(base64_str)
    
    # Process clip
    result = builder.process_video_clip(test_frames, clip_id=0)
    
    print("✅ Memory building test completed")
    return result

if __name__ == "__main__":
    test_memory_builder()
