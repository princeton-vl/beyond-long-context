"""
Weights & Biases logging utilities for video-language model metrics.
"""

from typing import Any, Dict, List, Optional
import json
import wandb


def calculate_video_metrics(model_metrics):
    """Calculate metrics statistics from model metrics."""
    if not model_metrics:
        return {}

    stats = {}

    # Calculate averages for different metric types
    if model_metrics.latency_add_video:
        stats['avg_latency_add_video'] = sum(model_metrics.latency_add_video) / len(model_metrics.latency_add_video)

    if model_metrics.latency_ask_question:
        stats['avg_latency_ask_question'] = sum(model_metrics.latency_ask_question) / len(model_metrics.latency_ask_question)

    if model_metrics.latency_add_text:
        stats['avg_latency_add_text'] = sum(model_metrics.latency_add_text) / len(model_metrics.latency_add_text)

    # Peak GPU memory stats
    all_peak_mem = (model_metrics.peak_gpu_mem_increase_add_video +
                   model_metrics.peak_gpu_mem_increase_add_text +
                   model_metrics.peak_gpu_mem_increase_ask_question)
    if all_peak_mem:
        stats['max_peak_gpu_mem_increase'] = max(all_peak_mem)
        stats['avg_peak_gpu_mem_increase'] = sum(all_peak_mem) / len(all_peak_mem)

    all_peak_abs = (
        model_metrics.peak_gpu_mem_absolute_add_video
        + model_metrics.peak_gpu_mem_absolute_add_text
        + model_metrics.peak_gpu_mem_absolute_ask_question
    )
    if all_peak_abs:
        stats['max_peak_gpu_mem_absolute'] = max(all_peak_abs)
        stats['avg_peak_gpu_mem_absolute'] = sum(all_peak_abs) / len(all_peak_abs)

    # GPU memory by operation type
    if model_metrics.peak_gpu_mem_increase_add_video:
        stats['avg_peak_gpu_mem_add_video'] = sum(model_metrics.peak_gpu_mem_increase_add_video) / len(model_metrics.peak_gpu_mem_increase_add_video)
        stats['max_peak_gpu_mem_add_video'] = max(model_metrics.peak_gpu_mem_increase_add_video)
    if model_metrics.peak_gpu_mem_absolute_add_video:
        stats['avg_peak_gpu_mem_absolute_add_video'] = sum(model_metrics.peak_gpu_mem_absolute_add_video) / len(model_metrics.peak_gpu_mem_absolute_add_video)
        stats['max_peak_gpu_mem_absolute_add_video'] = max(model_metrics.peak_gpu_mem_absolute_add_video)

    if model_metrics.peak_gpu_mem_increase_ask_question:
        stats['avg_peak_gpu_mem_ask_question'] = sum(model_metrics.peak_gpu_mem_increase_ask_question) / len(model_metrics.peak_gpu_mem_increase_ask_question)
        stats['max_peak_gpu_mem_ask_question'] = max(model_metrics.peak_gpu_mem_increase_ask_question)
    if model_metrics.peak_gpu_mem_absolute_ask_question:
        stats['avg_peak_gpu_mem_absolute_ask_question'] = sum(model_metrics.peak_gpu_mem_absolute_ask_question) / len(model_metrics.peak_gpu_mem_absolute_ask_question)
        stats['max_peak_gpu_mem_absolute_ask_question'] = max(model_metrics.peak_gpu_mem_absolute_ask_question)

    if model_metrics.peak_gpu_mem_increase_add_text:
        stats['avg_peak_gpu_mem_add_text'] = sum(model_metrics.peak_gpu_mem_increase_add_text) / len(model_metrics.peak_gpu_mem_increase_add_text)
        stats['max_peak_gpu_mem_add_text'] = max(model_metrics.peak_gpu_mem_increase_add_text)
    if model_metrics.peak_gpu_mem_absolute_add_text:
        stats['avg_peak_gpu_mem_absolute_add_text'] = sum(model_metrics.peak_gpu_mem_absolute_add_text) / len(model_metrics.peak_gpu_mem_absolute_add_text)
        stats['max_peak_gpu_mem_absolute_add_text'] = max(model_metrics.peak_gpu_mem_absolute_add_text)

    # State memory increments (handle potential type issues)
    if model_metrics.state_memory_floats and len(model_metrics.state_memory_floats) > 0:
        # Convert to floats in case there are tuples or other types
        clean_memory_values = []
        for val in model_metrics.state_memory_floats:
            if isinstance(val, (tuple, list)):
                clean_memory_values.append(float(val[0]) if val else 0.0)
            else:
                clean_memory_values.append(float(val))

        if len(clean_memory_values) > 1:
            increments = []
            for i in range(1, len(clean_memory_values)):
                increment = clean_memory_values[i] - clean_memory_values[i-1]
                increments.append(increment)

            if increments:
                stats['avg_state_memory_increase'] = sum(increments) / len(increments)
        elif len(clean_memory_values) == 1:
            stats['avg_state_memory_increase'] = clean_memory_values[0]

    # Operation-specific state deltas
    if hasattr(model_metrics, 'state_memory_delta_add_video') and model_metrics.state_memory_delta_add_video:
        stats['avg_state_delta_add_video'] = sum(model_metrics.state_memory_delta_add_video) / len(model_metrics.state_memory_delta_add_video)
        stats['max_state_delta_add_video'] = max(model_metrics.state_memory_delta_add_video)
    if hasattr(model_metrics, 'state_memory_delta_add_text') and model_metrics.state_memory_delta_add_text:
        stats['avg_state_delta_add_text'] = sum(model_metrics.state_memory_delta_add_text) / len(model_metrics.state_memory_delta_add_text)
        stats['max_state_delta_add_text'] = max(model_metrics.state_memory_delta_add_text)
    if hasattr(model_metrics, 'state_memory_delta_ask_question') and model_metrics.state_memory_delta_ask_question:
        stats['avg_state_delta_ask_question'] = sum(model_metrics.state_memory_delta_ask_question) / len(model_metrics.state_memory_delta_ask_question)
        stats['max_state_delta_ask_question'] = max(model_metrics.state_memory_delta_ask_question)

    return stats


class WandbLogger:
    """Weights & Biases logger for video-language model metrics."""

    def __init__(self, enabled: bool = True):
        """Initialize WandbLogger."""
        self.enabled = enabled
        self._question_table = None
        if self.enabled:
            try:
                wandb.init(project="streaming_video_models")
            except Exception as e:
                print(f"⚠️ Wandb initialization failed: {e}")
                self.enabled = False

    def log_video_completion(
        self,
        num_questions: int,
        video_accuracy: float,
        model_metrics: Optional[Any] = None,
        model_name: str = "unknown",
        extra_metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log completion of video processing."""
        if not self.enabled:
            return

        try:
            log_data = {
                'num_questions': num_questions,
                'video_accuracy': video_accuracy,
                'model_name': model_name
            }

            # Add model metrics if available
            if model_metrics:
                stats = calculate_video_metrics(model_metrics)
                log_data.update(stats)

            if extra_metrics:
                log_data.update(extra_metrics)

            wandb.log(log_data)

        except Exception as e:
            print(f"⚠️ Wandb logging failed: {e}")

    def log_final_results(
        self,
        total_videos: int,
        total_questions: int,
        overall_accuracy: float,
        idk_rate: float,
        model_name: str,
        extra_metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log final evaluation results."""
        if not self.enabled:
            return

        try:
            log_data = {
                'final_total_videos': total_videos,
                'final_total_questions': total_questions,
                'final_overall_accuracy': overall_accuracy,
                'final_idk_rate': idk_rate,
                'final_model_name': model_name,
            }
            if extra_metrics:
                for metric_key, metric_data in extra_metrics.items():
                    for field, value in metric_data.items():
                        if field in {'label', 'bins'} or value is None:
                            continue
                        log_data[f"final_{metric_key}_{field}"] = value
            wandb.log(log_data)
        except Exception as e:
            print(f"⚠️ Wandb final logging failed: {e}")

    def log_question_accuracy_scatter(self, samples: List[Dict[str, Any]]):
        """Log scatter plot of correctness over question timestamps."""
        if not self.enabled or not samples:
            return

        try:
            table = wandb.Table(
                columns=[
                    'timestamp_sec',
                    'is_correct',
                    'video_index',
                    'question_id',
                ]
            )
            for sample in samples:
                table.add_data(
                    float(sample.get('timestamp_sec', 0.0)),
                    int(1 if sample.get('is_correct') else 0),
                    sample.get('video_index', -1),
                    sample.get('question_id') or '',
                )
            scatter = wandb.plot.scatter(
                table,
                'timestamp_sec',
                'is_correct',
                title='Correctness vs Timestamp',
            )
            wandb.log({'question_correctness_scatter': scatter})
        except Exception as e:
            print(f"⚠️ Wandb scatter logging failed: {e}")

    def finish(self):
        """Finish wandb run."""
        if self.enabled:
            if self._question_table is not None:
                try:
                    wandb.log({'per_question_metrics': self._question_table})
                except Exception as exc:
                    print(f"⚠️ Wandb question table logging failed: {exc}")
            try:
                wandb.finish()
            except Exception as e:
                print(f"⚠️ Wandb finish failed: {e}")

    def log_question_record(self, record: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        try:
            if self._question_table is None:
                self._question_table = wandb.Table(
                columns=[
                    'video_index',
                    'question_id',
                    'question_order',
                    'timestamp_sec',
                    'is_correct',
                    'is_dont_know',
                    'latency_ask_question',
                    'peak_gpu_mem_ask_question',
                    'entropy_prefix_mean',
                    'prefix_match_fraction',
                    'correct_option_likelihood',
                    'frames_seen_before_question',
                    'response_token_count',
                    'token_breakdown_json',
                ]
            )

            self._question_table.add_data(
                record.get('video_index', -1),
                record.get('question_id'),
                record.get('question_order'),
                record.get('question_time'),
                int(1 if record.get('is_correct') else 0),
                int(1 if record.get('is_dont_know') else 0),
                record.get('latency_ask_question'),
                record.get('peak_gpu_mem_ask_question'),
                record.get('entropy_prefix_mean'),
                record.get('prefix_match_fraction'),
                record.get('correct_option_likelihood'),
                record.get('frames_seen_before_question'),
                record.get('response_token_count'),
                json.dumps(record.get('round_token_breakdown', [])) if record.get('round_token_breakdown') is not None else None,
            )
        except Exception as exc:
            print(f"⚠️ Wandb question logging failed: {exc}")
