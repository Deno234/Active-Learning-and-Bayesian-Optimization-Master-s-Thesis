from __future__ import annotations

import importlib
import importlib.util
from functools import lru_cache
import os
from pathlib import Path
import sys

from .paths import GENERATIVE_DIR, PREDICTIVE_CODE_DIR


def _ensure_path(path: Path) -> None:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def _prepend_path(path: Path) -> None:
    current_path = os.environ.get("PATH", "")
    path_entries = current_path.split(os.pathsep) if current_path else []
    path_str = str(path)
    if path_str not in path_entries:
        os.environ["PATH"] = path_str + os.pathsep + current_path


def _ensure_conda_library_bin_on_path() -> None:
    if os.name != "nt":
        return
    library_bin = Path(sys.prefix) / "Library" / "bin"
    if library_bin.exists():
        _prepend_path(library_bin)


def _cuda_bin_candidates() -> list[Path]:
    candidates: list[Path] = []

    for variable_name in ("CUDA_BIN_PATH", "CUDA_PATH", "CUDA_HOME", "CUDA_ROOT"):
        raw_value = os.environ.get(variable_name)
        if not raw_value:
            continue
        base_path = Path(raw_value)
        if base_path.name.lower() == "bin":
            candidates.append(base_path)
        else:
            candidates.append(base_path / "bin")

    for drive in ("C", "D", "E", "F", "G"):
        candidates.append(Path(f"{drive}:/CUDA/bin"))

    toolkit_root = Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA")
    if toolkit_root.exists():
        for child in sorted(toolkit_root.glob("v*/bin"), reverse=True):
            candidates.append(child)

    unique_candidates: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        candidate_str = str(candidate)
        if candidate_str not in seen:
            seen.add(candidate_str)
            unique_candidates.append(candidate)
    return unique_candidates


def _ensure_cuda_bin_on_path() -> Path | None:
    if os.name != "nt":
        return None
    for candidate in _cuda_bin_candidates():
        if (candidate / "ptxas.exe").exists():
            _prepend_path(candidate)
            return candidate
    return None


def _missing_predictive_runtime() -> list[str]:
    required = ["tensorflow", "keras", "pandas", "sklearn", "seqprops"]
    return [name for name in required if importlib.util.find_spec(name) is None]


@lru_cache(maxsize=1)
def configure_tensorflow_runtime() -> dict[str, object]:
    _ensure_conda_library_bin_on_path()
    ptxas_dir = _ensure_cuda_bin_on_path()

    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    configured_devices: list[str] = []
    for gpu in gpus:
        configured_devices.append(gpu.name)
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            # TensorFlow only allows memory growth to be set before device init.
            pass

    return {
        "tensorflow_version": tf.__version__,
        "built_with_cuda": bool(tf.test.is_built_with_cuda()),
        "gpu_devices": configured_devices,
        "ptxas_dir": str(ptxas_dir) if ptxas_dir is not None else "",
    }


def ensure_predictive_runtime() -> None:
    missing = _missing_predictive_runtime()
    if missing:
        missing_str = ", ".join(missing)
        raise RuntimeError(
            "The active-learning workflow needs the project runtime dependencies. "
            f"Missing: {missing_str}. Activate the project's conda environment first."
        )
    configure_tensorflow_runtime()


def load_predictive_modules():
    ensure_predictive_runtime()
    _ensure_path(PREDICTIVE_CODE_DIR)
    utils = importlib.import_module("utils")
    models = importlib.import_module("models")
    automate_training = importlib.import_module("automate_training")
    return utils, models, automate_training


def load_genetic_algorithm_class():
    _ensure_path(GENERATIVE_DIR)
    module = importlib.import_module("genetic_algorithm_library")
    return module.GeneticAlgorithm
