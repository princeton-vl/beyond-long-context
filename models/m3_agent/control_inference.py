"""
M3-Agent Control Inference Phase
Loads pre-built memory graphs and performs multi-round Q&A using vLLM.
"""

import re
import json
import os
import pickle
import signal
import time
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from .gpu_config import GPUConfig
from .video_graph import VideoGraph
from .video_graph_utils import load_video_graph, save_video_graph
from .text_embedding import TextEmbeddingBackend, load_api_config
from metrics.flops_calc import (
    m3_agent_control_flops,
    qwen3_embedding_0p6b_flops,
)
from utils.vllm_compat import ensure_vllm_disabled_tqdm_patch
from utils.paths import get_model_cache_dir
ensure_vllm_disabled_tqdm_patch()


CACHE_ROOT = get_model_cache_dir()
os.environ.setdefault("HF_HOME", CACHE_ROOT)
os.makedirs(CACHE_ROOT, exist_ok=True)

class ControlInference:
    """Handles Q&A inference using pre-built memory graphs."""
    
    def __init__(self, gpu_config: GPUConfig, enable_metrics: bool = False):
        self.gpu_config = gpu_config
        self.vllm_model = None
        self.tokenizer = None
        self.video_graphs = {}  # Cached VideoGraph objects
        self.flops = 0
        self.last_token_stats: List[Dict[str, int]] = []
        self.enable_metrics = enable_metrics
        config_path = os.path.join(os.path.dirname(__file__), "configs", "processing_config.json")
        api_config_path = os.path.join(os.path.dirname(__file__), "configs", "api_config.json")
        self.embedding_model = "text-embedding-3-large"
        self.embedding_device = "auto"
        self.max_control_rounds = 5
        if os.path.exists(config_path):
            try:
                processing_config = json.load(open(config_path))
                self.embedding_model = processing_config.get("embedding_model", self.embedding_model)
                self.max_control_rounds = processing_config.get("max_control_rounds", self.max_control_rounds)
                self.embedding_device = processing_config.get("embedding_device", self.embedding_device)
            except Exception as exc:
                print(f"Warning: failed to load control config: {exc}")

        # IMPORTANT: Initialize vLLM FIRST before any CUDA operations
        # This prevents multiprocessing issues when CUDA context already exists
        self._initialize_vllm()

        # Now initialize embedding backend after vLLM is ready
        if self.embedding_device == "cuda":
            try:
                qwen_allocation = self.gpu_config.get_allocation('qwen')
                self.embedding_device = f"cuda:{qwen_allocation['gpu']}"
            except Exception:
                self.embedding_device = "cuda"

        api_config = load_api_config(api_config_path)
        self.embedding_backend = TextEmbeddingBackend(
            self.embedding_model,
            api_config=api_config,
            target_dim=1536,
            device=self.embedding_device,
        )
    
    def _initialize_vllm(self):
        """Initialize vLLM model on allocated GPUs."""
        import os
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer

        vllm_allocation = self.gpu_config.get_allocation('vllm')
        model_name = "ByteDance-Seed/M3-Agent-Control"

        os.environ['HF_HOME'] = CACHE_ROOT
        os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
        existing_pythonpath = os.environ.get('PYTHONPATH')
        if existing_pythonpath:
            os.environ['PYTHONPATH'] = f"{repo_root}:{existing_pythonpath}"
        else:
            os.environ['PYTHONPATH'] = repo_root

        # Configure Ray to use specific GPUs for vLLM
        allocated_gpu_ids = vllm_allocation['gpus']
        gpu_ids_str = ','.join(map(str, allocated_gpu_ids))

        # Initialize Ray before vLLM does, with runtime_env to exclude large files
        # This prevents the "Package size exceeds 512MB" error
        import ray
        if not ray.is_initialized():
            # Define excludes for working directory to reduce package size
            excludes = [
                'wandb/*', 'logs/*', 'slurm/*', 'data/*',
                '*.mp4', '*.avi', '*.mov', '*.mkv', '*.webm',
                '*.pth', '*.bin', '*.safetensors', '*.ckpt', '*.pt',
                'cache/*', '.cache/*', '__pycache__/*', '*.pyc',
                'outputs/*', 'runs/*', '*.log', '*.out', '*.err',
                '*.tar.gz', '*.zip', '*.jsonl', 'old/*', 'old_docs_and_tests_*/*',
                'results/*', 'tests/*', 'external/*', '.venv/*', '.git/*'
            ]
            runtime_env = {
                "working_dir": repo_root,
                "excludes": excludes,
            }
            # Configure Ray with specific GPU IDs
            ray.init(
                runtime_env=runtime_env,
                ignore_reinit_error=True,
                num_gpus=len(allocated_gpu_ids),
            )

        # Use vLLM with dynamically allocated tensor parallel size
        print(f"Initializing vLLM on GPUs {allocated_gpu_ids} (TP={vllm_allocation['tensor_parallel_size']})")

        # Adjust GPU memory utilization based on whether components are sharing GPUs
        # Check if Qwen or InsightFace share any GPU with vLLM
        vllm_gpus = set(vllm_allocation['gpus'])
        qwen_gpu = self.gpu_config.allocations.get('qwen', {}).get('gpu')
        insightface_gpu = self.gpu_config.allocations.get('insightface', {}).get('gpu')

        sharing_gpu = (
            (qwen_gpu is not None and qwen_gpu in vllm_gpus) or
            (insightface_gpu is not None and insightface_gpu in vllm_gpus)
        )

        # If sharing single GPU with Qwen (~30GB) + InsightFace (~2GB), reduce vLLM to ~50%
        # Otherwise use 85% as normal
        gpu_mem_util = 0.5 if (sharing_gpu and vllm_allocation['tensor_parallel_size'] == 1) else 0.85

        print(f"  GPU memory utilization: {gpu_mem_util:.1%} (sharing={sharing_gpu})")

        try:
            self.vllm_model = LLM(
                model=model_name,
                tensor_parallel_size=vllm_allocation['tensor_parallel_size'],
                trust_remote_code=True,
                gpu_memory_utilization=gpu_mem_util,
                disable_log_stats=True,
            )
        except Exception as e:
            print(f"❌ vLLM M3-Agent-Control: Failed to initialize on GPUs {vllm_allocation['gpus']}")
            print(f"   Error: {str(e)}")
            raise

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=CACHE_ROOT,
            trust_remote_code=True,
        )
        
        self.sampling_params = SamplingParams(
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            max_tokens=1024
        )
    
    def _check_gpu_sharing(self, vllm_gpu_ids):
        """Check if other models are sharing vLLM GPUs."""
        other_allocations = [
            self.gpu_config.allocations.get('qwen', {}).get('gpu'),
            self.gpu_config.allocations.get('insightface', {}).get('gpu')
        ]
        return any(gpu_id in vllm_gpu_ids for gpu_id in other_allocations if gpu_id is not None)

    def cleanup_vllm(self):
        """
        Clean up vLLM and Ray processes.

        This method ensures all vLLM worker processes and Ray actors are properly
        terminated to prevent zombie processes from holding GPU memory.
        """
        import signal
        import psutil
        import time

        print("Cleaning up vLLM and Ray processes...")

        # Get GPU IDs used by vLLM
        try:
            vllm_allocation = self.gpu_config.get_allocation('vllm')
            gpu_ids = vllm_allocation['gpus']
            gpu_ids_str = ','.join(map(str, gpu_ids))
        except Exception:
            gpu_ids = []
            gpu_ids_str = ""

        # Shutdown Ray first (this should shutdown vLLM workers)
        try:
            import ray
            if ray.is_initialized():
                print("  Shutting down Ray...")
                ray.shutdown()
                print("  Ray shutdown complete")
        except Exception as e:
            print(f"  Warning: Ray shutdown failed: {e}")

        # Additional cleanup: kill any remaining vLLM/Ray processes
        try:
            current_pid = os.getpid()
            current_process = psutil.Process(current_pid)

            # Get all child processes
            children = current_process.children(recursive=True)

            if children:
                print(f"  Found {len(children)} child processes")

                # Try graceful termination first (SIGTERM)
                for child in children:
                    try:
                        if child.is_running():
                            cmdline = ' '.join(child.cmdline()[:3])  # First 3 args
                            if any(keyword in cmdline for keyword in ['vllm', 'ray', 'python']):
                                print(f"    Terminating PID {child.pid} ({child.name()})")
                                child.terminate()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass

                # Wait for graceful shutdown
                print("  Waiting 3s for graceful shutdown...")
                time.sleep(3)

                # Force kill remaining processes (SIGKILL)
                for child in children:
                    try:
                        if child.is_running():
                            print(f"    Force killing PID {child.pid} ({child.name()})")
                            child.kill()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
        except Exception as e:
            print(f"  Warning: Child process cleanup failed: {e}")

        # GPU-specific cleanup: kill processes on allocated GPUs
        if gpu_ids:
            try:
                import subprocess
                print(f"  Cleaning up processes on GPUs: {gpu_ids_str}")

                for gpu_id in gpu_ids:
                    try:
                        # Get PIDs of processes using this GPU
                        result = subprocess.run(
                            ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader', f'--id={gpu_id}'],
                            capture_output=True,
                            text=True,
                            timeout=5
                        )

                        if result.returncode == 0:
                            pids = [int(pid.strip()) for pid in result.stdout.strip().split('\n') if pid.strip()]

                            for pid in pids:
                                try:
                                    proc = psutil.Process(pid)
                                    cmdline = ' '.join(proc.cmdline())

                                    # Only kill vLLM/Ray/Python processes, not everything on the GPU
                                    if any(keyword in cmdline for keyword in ['vllm', 'ray', 'python']):
                                        print(f"    Force killing process {pid} on GPU {gpu_id}")
                                        proc.kill()
                                except (psutil.NoSuchProcess, psutil.AccessDenied):
                                    pass
                    except Exception as e:
                        print(f"    Warning: GPU {gpu_id} cleanup failed: {e}")

            except Exception as e:
                print(f"  Warning: GPU-specific cleanup failed: {e}")

        # Clear CUDA cache
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                print("  CUDA cache cleared")
        except Exception as e:
            print(f"  Warning: CUDA cache clear failed: {e}")

        print("vLLM cleanup complete")

    def __del__(self):
        """Cleanup on object destruction."""
        try:
            self.cleanup_vllm()
        except Exception:
            pass  # Silent cleanup on deletion

    def load_video_graph(self, video_id: str, video_graph: VideoGraph):
        """Load VideoGraph for a video."""
        self.video_graphs[video_id] = video_graph
        stats = video_graph.get_stats()
        print(f"Loaded VideoGraph for video {video_id}: {stats}")
    
    def answer_question(self, question: str, main_video_id: str,
                       option_videos: List[Dict[str, str]], max_rounds: Optional[int] = None,
                       max_tokens: int = 1024) -> str:
        """Answer question using multi-round search and inference."""

        self.last_token_stats = []

        # Create sampling params with dynamic max_tokens
        sampling_params = SamplingParams(
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            max_tokens=max_tokens
        )

        # Load main video graph
        if main_video_id not in self.video_graphs:
            raise RuntimeError("Main video graph not loaded!")
        
        # Create enhanced question with options (matching original M3-Agent implementation)
        enhanced_question = question
        if option_videos:
            print(f"Building option descriptions for {len(option_videos)} options")
            option_descriptions = []
            for option in sorted(option_videos, key=lambda x: x.get("option_index", 0)):
                # Use mem_path from option data, matching original answer_control.py line 271
                # Fallback to in-memory graph if mem_path not available
                if "mem_path" in option:
                    option_desc = self._get_memory_description(option["mem_path"])
                else:
                    option_desc = self._get_memory_description_from_graph(option["video_id"])
                
                option_text = f"Option {option['option_index']}: {option_desc}"
                option_descriptions.append(option_text)
                print(f"Option {option['option_index']}: {option_desc[:100]}...")
            
            enhanced_question += "\n\nOptions:\n" + "\n".join(option_descriptions)
            print(f"Enhanced question: {len(enhanced_question)} chars")
        
        # Initialize conversation
        system_prompt = self._get_system_prompt(enhanced_question)
        conversations = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Searched knowledge: {}"}
        ]
        
        current_clips = []
        
        if max_rounds is None:
            max_rounds = self.max_control_rounds

        # Multi-round inference loop
        for round_idx in range(max_rounds):
            print(f"Inference round {round_idx + 1}/{max_rounds}")
            
            # Add instruction for this round
            instruction = self._get_instruction(round_idx, max_rounds)
            conversations[-1]["content"] += instruction
            
            # Generate response using vLLM
            response = self._generate_response(conversations, sampling_params)
            conversations.append({"role": "assistant", "content": response})
            
            # Parse action from response - handle last round specially
            try:
                action, content = self._parse_action(response)
                print(f"  Action: {action}")
                if action == "Search":
                    print(f"  Search Query: {content}")
                elif action == "Answer":
                    print(f"  Answer Content: {content}")
                    print(f"  Full Response: {response}")
            except RuntimeError as e:
                if round_idx == max_rounds - 1:
                    # Last round - return "no answer" instead of failing
                    print(f"  Could not parse response on final round, returning no answer")
                    print(f"  Response preview: {response[:200]}...")
                    return "no answer"
                else:
                    # Not last round - continue to next round with empty search
                    print(f"  Could not parse response, moving to next round")
                    print(f"  Response preview: {response[:200]}...")
                    # Add empty search result to continue the conversation
                    search_content = "Searched knowledge: {}\n(The previous response could not be parsed. Please try again with proper Action: [Search] or Action: [Answer] format.)"
                    conversations.append({"role": "user", "content": search_content})
                    continue
            
            if action == "Answer":
                print(f"Final answer: {content}")
                # Store full response for potential future use
                self.last_full_response = response
                return content
            elif action == "Search":
                # Perform search exactly like original M3-Agent
                video_graph = self.video_graphs[main_video_id]
                
                # CRITICAL: Call refresh_equivalences like original
                video_graph.refresh_equivalences()
                
                new_memories = {}
                if content:
                    if "character id" in content:
                        # Special character ID search (like original)
                        print(f"  → Character ID search with topk=20")
                        search_results, current_clips, clip_scores = self.search(
                            video_graph, content, [], topk=20, mem_wise=True, before_clip=None
                        )
                        print(f"  → Found {len(search_results)} character memories")
                        new_memories.update(search_results)
                    else:
                        # Normal search with original parameters
                        print(f"  → Normal search with topk=2, threshold=0.5")
                        search_results, current_clips, clip_scores = self.search(
                            video_graph, content, current_clips, topk=2, threshold=0.5, mem_wise=False
                        )
                        print(f"  → Found {len(search_results)} memories from {len(current_clips)} clips")
                        print(f"  → Clip scores: {dict(list(clip_scores.items())[:5])}")  # Show top 5 clips
                        print(f"  → Search result keys: {list(search_results.keys())}")
                        if search_results:
                            # Show first few memory contents
                            first_key = list(search_results.keys())[0]
                            sample_content = str(search_results[first_key])
                            print(f"  → Sample memory content: {sample_content}")
                        current_clips = current_clips  # Update current clips like original
                        new_memories.update(search_results)
                
                # Format exactly like original
                search_content = "Searched knowledge: " + json.dumps(new_memories, ensure_ascii=False).encode("utf-8", "ignore").decode("utf-8")
                if len(new_memories) == 0:
                    search_content += "\n(The search result is empty. Please try searching from another perspective.)"
                    print(f"  → ❌ Empty search results!")
                else:
                    print(f"  → ✅ Found memories: {list(new_memories.keys())}")  # Show all memory keys
                    
                print(f"  → Search content length: {len(search_content)}")
                conversations.append({"role": "user", "content": search_content})
            else:
                # Fallback for unclear responses
                conversations.append({"role": "user", "content": "Searched knowledge: {}"})
        
        print(f"❌ Unable to determine answer after {max_rounds} rounds of search")
        print(f"❌ Final conversation had {len(conversations)} messages")
        return "Unable to determine answer after maximum rounds."
    
    def _get_system_prompt(self, question: str) -> str:
        """Get system prompt for M3-Agent - exact match to original."""
        return f"""You are given a question and some relevant knowledge. Your task is to reason about whether the provided knowledge is sufficient to answer the question. If it is sufficient, output [Answer] followed by the answer. If it is not sufficient, output [Search] and generate a query that will be encoded into embeddings for a vector similarity search. The query will help retrieve additional information from a memory bank.

Question: {question}"""
    
    def _get_instruction(self, round_idx: int, max_rounds: int) -> str:
        """Get instruction for current round - exact match to original."""
        instruction = """

Output the answer in the format:
Action: [Answer] or [Search]
Content: {content}

If the answer cannot be derived yet, the {content} should be a single search query that would help retrieve the missing information. The search {content} needs to be different from the previous.
You can get the mapping relationship between character ID and name by using search query such as: "What is the name of <character_{i}>" or "What is the character id of {name}".
After obtaining the mapping, it is best to use character ID instead of name for searching.
If the answer can be derived from the provided knowledge, the {content} is the specific answer to the question. Only name can appear in the answer, not character ID like <character_{i}>."""
        
        # CRITICAL: Force answer in final round (like original M3-Agent)
        if round_idx == max_rounds - 1:
            instruction += "\n(The Action of this round must be [Answer]. If there is insufficient information, you can make reasonable guesses.)"
        
        return instruction
    
    def _generate_response(self, conversations: List[Dict[str, str]], sampling_params=None) -> str:
        """Generate response using vLLM."""
        # Use default sampling params if not provided
        if sampling_params is None:
            sampling_params = self.sampling_params

        # Use apply_chat_template with enable_thinking=True (crucial for <think> tags)
        prompt_tokens = self.tokenizer.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=True
        )

        # Generate using vLLM
        outputs = self.vllm_model.generate(
            prompts=[{"prompt_token_ids": prompt_tokens}],
            sampling_params=sampling_params,
            use_tqdm=False
        )

        if outputs and len(outputs) > 0 and outputs[0].outputs and len(outputs[0].outputs) > 0:
            generated_text = outputs[0].outputs[0].text.strip()

            input_tokens = len(prompt_tokens)
            output_tokens = len(self.tokenizer.encode(generated_text))
            self.last_token_stats.append(
                {
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                }
            )

            if self.enable_metrics:
                inference_flops = m3_agent_control_flops(
                    lang_prompt_len=input_tokens,
                    num_generated=output_tokens,
                    do_backward=False
                )
                total_flops = inference_flops.get('total_flops', 0)
                self.flops += total_flops

            return generated_text

        raise RuntimeError("Failed to generate response from vLLM")

    def get_and_reset_flops(self) -> float:
        """Get accumulated FLOPS and reset counter."""
        flops = self.flops
        self.flops = 0
        return flops

    def get_last_token_stats(self) -> List[Dict[str, int]]:
        return [dict(entry) for entry in self.last_token_stats]

    def _parse_action(self, response: str) -> tuple[str, str]:
        """Parse action from response with robust fallback handling."""
        original_response = response
        
        # Handle thinking tags more robustly
        if "</think>" in response:
            response = response.split("</think>")[-1]
        elif "<think>" in response and "</think>" not in response:
            # Incomplete thinking - try to extract content after <think>
            parts = response.split("<think>")
            if len(parts) > 1:
                # Look for action pattern in the thinking content
                thinking_content = parts[-1]
                if "Action:" in thinking_content or "[Answer]" in thinking_content or "[Search]" in thinking_content:
                    response = thinking_content
        
        # Clean up the response
        response = response.strip()
        
        # Look for action pattern - primary format
        pattern = r"Action: \[(.*?)\].*?Content: (.*)"
        match = re.search(pattern, response, re.DOTALL)
        
        if match:
            action = match.group(1).strip()
            content = match.group(2).strip()
            return action, content
        
        # Fallback: Look for direct [Answer] or [Search] patterns
        answer_pattern = r"\[Answer\][:\s]*(.*)"
        search_pattern = r"\[Search\][:\s]*(.*)"
        
        answer_match = re.search(answer_pattern, response, re.DOTALL)
        if answer_match:
            return "Answer", answer_match.group(1).strip()
            
        search_match = re.search(search_pattern, response, re.DOTALL)
        if search_match:
            return "Search", search_match.group(1).strip()
        
        # Final fallback: If response contains answer-like content, assume it's an answer
        if any(pattern in response for pattern in ["{0}", "{1}", "{2}", "{3}", "{4}"]):
            return "Answer", response.strip()
        
        # If all else fails, this should cause an error (no fallback)
        raise RuntimeError(f"Could not parse action from response: {original_response[:200]}...")
    
    def search(self, video_graph, query: str, current_clips: List, topk: int = 5,
              mode: str = 'max', threshold: float = 0.5, mem_wise: bool = False,
              before_clip: int = None, episodic_only: bool = False) -> Tuple[Dict, List, Dict]:
        """EXACT search logic from retrieve.py lines 237-275 with character translation."""

        # Step 1: Expand queries using character mappings (EXACT from retrieve.py)
        queries = self.back_translate(video_graph, [query])
        if len(queries) > 100:
            import random
            print(f"Anomaly detected from query: {query}, randomly sample 100 translated queries")
            queries = random.sample(queries, 100)

        # Step 2: Get related nodes (EXACT from retrieve.py)
        related_nodes = self.get_related_nodes(video_graph, query)

        # Step 3: Get embeddings for all query variants
        query_embeddings = []
        for q in queries:
            emb = self._get_query_embedding(q)
            if emb:
                query_embeddings.append(emb)

        if not query_embeddings:
            raise RuntimeError(f"Failed to embed any query variants for '{query}'")

        # Step 4: Search using VideoGraph with ALL query embeddings (EXACT from retrieve.py line 106)
        nodes = video_graph.search_text_nodes(query_embeddings, related_nodes, mode='max')

        # Step 5: Calculate clip scores (EXACT from retrieve.py lines 109-135)
        full_clip_scores = {}
        for node_id, node_score in nodes:
            clip_id = video_graph.nodes[node_id].metadata['timestamp']
            if clip_id not in full_clip_scores:
                full_clip_scores[clip_id] = []
            full_clip_scores[clip_id].append(node_score)

        clip_scores = {}
        for clip_id, scores in full_clip_scores.items():
            if mode == 'sum':
                clip_score = sum(scores)
            elif mode == 'max':
                clip_score = max(scores)
            elif mode == 'mean':
                clip_score = np.mean(scores)
            else:
                clip_score = max(scores)
            clip_scores[clip_id] = clip_score

        # Step 6: Get top clips (EXACT from retrieve.py lines 130-136)
        sorted_clips = sorted(clip_scores.items(), key=lambda x: x[1], reverse=True)
        if before_clip is not None:
            top_clips = [clip_id for clip_id, score in sorted_clips if score >= threshold and clip_id <= before_clip][:topk]
        else:
            top_clips = [clip_id for clip_id, score in sorted_clips if score >= threshold][:topk]

        # Step 7: Handle mem_wise retrieval (EXACT from retrieve.py lines 240-258)
        if mem_wise:
            new_memories = {}
            top_nodes_num = 0
            for top_node, _ in nodes:
                clip_id = video_graph.nodes[top_node].metadata['timestamp']
                if before_clip is not None and clip_id > before_clip:
                    continue
                if clip_id not in new_memories:
                    new_memories[clip_id] = []
                contents = video_graph.nodes[top_node].metadata['contents']
                translated_contents = self.translate(video_graph, contents)  # Apply translation
                new_memories[clip_id].extend(translated_contents)
                top_nodes_num += len(translated_contents)
                if top_nodes_num >= topk:
                    break

            new_memories = dict(sorted(new_memories.items(), key=lambda x: x[0]))
            new_memories = {f"CLIP_{k}": v for k, v in new_memories.items() if len(v) > 0}
            return new_memories, current_clips, clip_scores

        # Step 8: Handle clip-wise retrieval (EXACT from retrieve.py lines 260-275)
        new_clips = [top_clip for top_clip in top_clips if top_clip not in current_clips]
        new_memories = {}
        current_clips.extend(new_clips)

        for new_clip in new_clips:
            if new_clip not in video_graph.text_nodes_by_clip:
                new_memories[new_clip] = [f"CLIP_{new_clip} not found in memory bank, please search for other information"]
            else:
                related_nodes = video_graph.text_nodes_by_clip[new_clip]
                contents = [video_graph.nodes[node_id].metadata['contents'][0] for node_id in related_nodes
                           if (not episodic_only or video_graph.nodes[node_id].type != "semantic")]
                translated_contents = self.translate(video_graph, contents)  # Apply translation
                new_memories[new_clip] = translated_contents

        new_memories = dict(sorted(new_memories.items(), key=lambda x: x[0]))
        new_memories = {f"CLIP_{k}": v for k, v in new_memories.items()}

        return new_memories, current_clips, clip_scores
    
    def _search_memories(self, video_id: str, query: str, current_clips: List) -> Dict:
        """Search memories using VideoGraph and embedding similarity."""
        if video_id not in self.video_graphs:
            return {}
        
        video_graph = self.video_graphs[video_id]
        
        # Get query embedding
        query_embedding = self._get_query_embedding(query)
        if not query_embedding:
            # Fallback to first few clips
            search_results = {}
            for clip_id in list(video_graph.text_nodes_by_clip.keys())[:3]:
                memories = video_graph.get_clip_memories(clip_id)
                if memories:
                    search_results[f"CLIP_{clip_id}"] = memories
            return search_results
        
        # Search using VideoGraph's text node search
        top_nodes = video_graph.search_text_nodes(query_embedding, topk=5, threshold=0.3)
        
        # Group results by clip
        search_results = {}
        for node_id, similarity in top_nodes:
            node = video_graph.nodes[node_id]
            clip_id = node.metadata.get('timestamp', 0)
            clip_key = f"CLIP_{clip_id}"
            
            if clip_key not in search_results:
                search_results[clip_key] = []
            
            contents = node.metadata.get('contents', [])
            search_results[clip_key].extend(contents)
        
        return search_results
    
    def _get_query_embedding(self, query: str) -> Optional[List[float]]:
        """Get embedding for search query using configured backend."""
        if not self.embedding_backend:
            return None

        result = self.embedding_backend.embed_text(query, role="query")
        if not result.vectors:
            return None

        embedding = result.vectors[0]

        if self.enable_metrics:
            provider = result.provider.lower()
            embed_flops = 0.0
            if provider == "qwen":
                token_lengths = self.embedding_backend.token_lengths([query], role="query")
                if token_lengths:
                    seq_len = token_lengths[0]
                    if seq_len > 0:
                        embed_flops = qwen3_embedding_0p6b_flops(1, seq_len)
            self.flops += embed_flops

        return embedding
    
    def _get_memory_description(self, mem_path: str) -> str:
        """
        Extract structured description from memory graph with proper labeling and ordering.
        
        Based on the original M3-Agent implementation from answer_control.py lines 140-218.
        
        Returns memories organized by:
        - Temporal order (clip_id/timestamp)  
        - Memory type (episodic vs semantic)
        - Node metadata (type, timestamp, content)
        """
        try:
            # Load video graph from pickle file (like original implementation)
            if not os.path.exists(mem_path):
                print(f"Warning: Memory file not found: {mem_path}")
                return f"[No memory file available for {os.path.basename(mem_path)}]"
            
            with open(mem_path, "rb") as f:
                mem_node = pickle.load(f)
            
            # Collect all memories with metadata (original approach)
            memory_entries = []
            for node_id, node in mem_node.nodes.items():
                if not hasattr(node, 'metadata') or 'contents' not in node.metadata:
                    continue
                    
                contents = node.metadata.get('contents', [])
                if isinstance(contents, list):
                    content_list = [str(content) for content in contents]
                elif isinstance(contents, str):
                    content_list = [str(contents)]
                else:
                    continue
                
                # Extract metadata
                timestamp = node.metadata.get('timestamp', 0)
                node_type = getattr(node, 'type', 'unknown')
                
                for content in content_list:
                    memory_entries.append({
                        'timestamp': timestamp,
                        'type': node_type,
                        'node_id': node_id,
                        'content': content
                    })
            
            if not memory_entries:
                print(f"Warning: No content found in memory graph nodes: {mem_path}")
                return f"[No memory content available for {os.path.basename(mem_path)}]"
            
            # Sort by timestamp first, then by type (episodic before semantic)  
            type_order = {'episodic': 0, 'semantic': 1, 'img': 2, 'voice': 3, 'unknown': 4}
            try:
                memory_entries.sort(key=lambda x: (x['timestamp'], type_order.get(x['type'], 4), x['node_id']))
            except (KeyError, TypeError) as e:
                print(f"Warning: Error sorting memories for {mem_path}: {e}")
                # Fallback: sort by node_id only
                memory_entries.sort(key=lambda x: x['node_id'])
            
            # Format structured output with separate numbering for each type (original format)
            formatted_memories = []
            current_timestamp = None
            type_counters = {}  # Track numbering for each memory type per timestamp
            
            for entry in memory_entries:
                # Add timestamp header when it changes
                if entry['timestamp'] != current_timestamp:
                    current_timestamp = entry['timestamp']
                    type_counters = {}  # Reset counters for new timestamp
                    if len(formatted_memories) > 0:  # Not first timestamp
                        formatted_memories.append("")  # Blank line separator
                    formatted_memories.append(f"[CLIP_{current_timestamp}]")
                
                # Increment counter for this memory type
                mem_type = entry['type'].upper()
                if mem_type not in type_counters:
                    type_counters[mem_type] = 0
                type_counters[mem_type] += 1
                
                # Format memory with type label and separate numbering
                memory_line = f"  [{mem_type}_{type_counters[mem_type]}]: {entry['content']}"
                formatted_memories.append(memory_line)
            
            return "\n".join(formatted_memories)
            
        except Exception as e:
            print(f"Error loading memory graph {mem_path}: {e}")
            raise RuntimeError(f"Failed to load memory graph {mem_path}: {e}") from e
    
    def _get_memory_description_from_graph(self, video_id: str) -> str:
        """Get description from in-memory VideoGraph matching original format exactly."""
        if video_id not in self.video_graphs:
            raise RuntimeError(f"Video graph for {video_id} not found! Available: {list(self.video_graphs.keys())}")
        
        video_graph = self.video_graphs[video_id]
        
        # Collect all memories with metadata (matching original answer_control.py lines 153-176)
        memory_entries = []
        for node_id, node in video_graph.nodes.items():
            if not hasattr(node, 'metadata') or 'contents' not in node.metadata:
                continue
                
            contents = node.metadata.get('contents', [])
            if isinstance(contents, list):
                content_list = [str(content) for content in contents]
            elif isinstance(contents, str):
                content_list = [str(contents)]
            else:
                continue
            
            # Extract metadata (matching original lines 167-169)
            timestamp = node.metadata.get('timestamp', 0)
            node_type = getattr(node, 'type', 'unknown')
            
            for content in content_list:
                memory_entries.append({
                    'timestamp': timestamp,
                    'type': node_type,
                    'node_id': node_id,
                    'content': content
                })
        
        if not memory_entries:
            raise RuntimeError(f"No memory entries found in video graph {video_id}! Graph has {len(video_graph.nodes)} nodes")
        
        # Sort by timestamp first, then by type (matching original lines 183-189)
        type_order = {'episodic': 0, 'semantic': 1, 'img': 2, 'voice': 3, 'unknown': 4}
        try:
            memory_entries.sort(key=lambda x: (x['timestamp'], type_order.get(x['type'], 4), x['node_id']))
        except (KeyError, TypeError):
            # Fallback: sort by node_id only
            memory_entries.sort(key=lambda x: x['node_id'])
        
        # Format structured output with separate numbering for each type (matching original lines 192-214)
        formatted_memories = []
        current_timestamp = None
        type_counters = {}  # Track numbering for each memory type per timestamp
        
        for entry in memory_entries:
            # Add timestamp header when it changes (matching original lines 197-203)
            if entry['timestamp'] != current_timestamp:
                current_timestamp = entry['timestamp']
                type_counters = {}  # Reset counters for new timestamp
                if len(formatted_memories) > 0:  # Not first timestamp
                    formatted_memories.append("")  # Blank line separator
                formatted_memories.append(f"[CLIP_{current_timestamp}]")
            
            # Increment counter for this memory type (matching original lines 205-209)
            mem_type = entry['type'].upper()
            if mem_type not in type_counters:
                type_counters[mem_type] = 0
            type_counters[mem_type] += 1
            
            # Format memory with type label and separate numbering (matching original line 212)
            memory_line = f"  [{mem_type}_{type_counters[mem_type]}]: {entry['content']}"
            formatted_memories.append(memory_line)
        
        return "\n".join(formatted_memories)

    def translate(self, video_graph, memories):
        """Convert entity IDs to human names - EXACT from retrieve.py lines 35-48."""
        if not hasattr(video_graph, 'reverse_character_mappings'):
            return memories

        from .memory_builder import parse_video_caption
        new_memories = []
        for memory in memories:
            if memory.lower().startswith("equivalence: "):
                continue
            new_memory = memory
            entities = parse_video_caption(video_graph, memory)
            entities = list(set(entities))  # Remove duplicates
            for entity_type, entity_id in entities:
                entity_str = f"{entity_type}_{entity_id}"
                if entity_str in video_graph.reverse_character_mappings:
                    new_memory = new_memory.replace(entity_str, video_graph.reverse_character_mappings[entity_str])
            new_memories.append(new_memory)
        return new_memories

    def back_translate(self, video_graph, queries):
        """Expand queries with character mappings - EXACT from retrieve.py lines 50-73."""
        if not hasattr(video_graph, 'character_mappings'):
            return queries

        from .memory_builder import parse_video_caption
        translated_queries = []
        for query in queries:
            entities = parse_video_caption(video_graph, query)
            entities = list(set(entities))
            to_be_translated = [query]
            for entity_type, entity_id in entities:
                entity_str = f"{entity_type}_{entity_id}"
                if entity_str in video_graph.character_mappings:
                    mappings = video_graph.character_mappings[entity_str]

                    new_queries = []
                    for mapping in mappings:
                        for partially_translated in to_be_translated:
                            new_query = partially_translated.replace(entity_str, mapping)
                            new_queries.append(new_query)

                    to_be_translated = new_queries

            translated_queries.extend(to_be_translated)

        # Add safety limit to prevent API explosion
        if len(translated_queries) > 100:
            print(f"WARNING: Query expansion created {len(translated_queries)} variants, limiting to 100")
            translated_queries = translated_queries[:100]

        return translated_queries

    def get_related_nodes(self, video_graph, query):
        """Get related nodes for query - EXACT from retrieve.py lines 138-150."""
        from .memory_builder import parse_video_caption
        related_nodes = []
        entities = parse_video_caption(video_graph, query)
        for entity_type, entity_id in entities:
            entity_str = f"{entity_type}_{entity_id}"
            if not (hasattr(video_graph, 'character_mappings') and entity_str in video_graph.character_mappings) and \
               not (hasattr(video_graph, 'reverse_character_mappings') and entity_str in video_graph.reverse_character_mappings):
                continue
            if entity_type == "character":
                if hasattr(video_graph, 'character_mappings') and entity_str in video_graph.character_mappings:
                    related_nodes.extend([int(node.split("_")[1]) for node in video_graph.character_mappings[entity_str]])
            else:
                related_nodes.append(entity_id)
        return list(set(related_nodes))

def test_control_inference():
    """Test control inference with mock data."""
    print("Testing Control Inference...")
    
    # This would normally require GPU, so just test class creation
    try:
        # Mock GPU config for testing
        class MockGPUConfig:
            def get_allocation(self, component):
                return {
                    'gpus': [0, 1],
                    'tensor_parallel_size': 2
                }
        
        # Test memory description function with non-existent file
        ci = ControlInference.__new__(ControlInference)  # Create without __init__
        test_result = ci._get_memory_description("/non/existent/path.pkl")
        assert "No memory file available" in test_result
        print("✅ Memory description function handles missing files correctly")
        
        # Skip actual vLLM initialization for testing
        print("✅ Control inference architecture created")
        print("✅ (GPU initialization skipped in test environment)")
        
    except Exception as e:
        print(f"❌ Control inference test failed: {e}")

if __name__ == "__main__":
    test_control_inference()
