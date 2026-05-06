from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Optional, Sequence, Tuple

import yaml


class TemplateOverrideManager:
    """Utility to materialize temporary templates with overrides and clean them up."""

    def __init__(self) -> None:
        self._tmp_dirs: list[Path] = []

    def build(
        self,
        base_template: Path,
        *,
        step_range: Optional[Sequence[float]] = None,
        fps: Optional[int] = None,
        hide_question_text: bool = False,
    ) -> Path:
        data = yaml.safe_load(base_template.read_text())
        if data is None:
            data = {}
        if step_range is not None:
            timing = data.setdefault("timing", {})
            timing["step_duration_range"] = [float(step_range[0]), float(step_range[1])]
        if fps is not None:
            render = data.setdefault("render", {})
            render["fps"] = int(fps)
        if hide_question_text:
            data.setdefault("scene", {})["show_hud_label"] = False
            if "question_text" in data:
                data["question_text"] = []
            if "question_overlay" in data:
                data["question_overlay"] = []
        tmp_dir = Path(tempfile.mkdtemp(prefix="template_override_"))
        self._tmp_dirs.append(tmp_dir)
        out_path = tmp_dir / Path(base_template).name
        out_path.write_text(yaml.safe_dump(data, sort_keys=False))
        return out_path

    def cleanup(self) -> None:
        for tmp in self._tmp_dirs:
            shutil.rmtree(tmp, ignore_errors=True)
        self._tmp_dirs.clear()
