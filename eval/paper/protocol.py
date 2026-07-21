"""Load, validate, and fingerprint the immutable paper-evaluation protocol."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Sequence

import yaml

from . import SCHEMA_VERSION


SHA_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SUPPORTED_SUITES = ("internal", "deepsdo", "astrovlbench")
COMMON_GENERATION_FILES = (
    "scripts/paper_eval_worker.py",
    "eval/paper/model_backends.py",
    "decode_utils.py",
)
GENERATION_SYMBOLS = {
    "eval/paper/artifacts.py": (
        "ValidationReport",
        "PredictionStore",
        "build_attempt",
        "read_jsonl",
        "record_fingerprint",
        "technical_status",
        "text_hash",
        "utc_now",
        "write_json_atomic",
        "write_jsonl_atomic",
    ),
    "eval/paper/assets.py": (
        "AssetRegistry",
        "download_snapshot",
        "safe_extract_zip",
        "snapshot_commit",
        "verify_file",
        "verify_snapshot_revision",
        "zip_member_sha256",
    ),
}
BACKEND_GENERATION_FILES = {
    "astraq": (
        "inference.py",
        "data/image_processing.py",
        "training/checkpoint.py",
        "vlm_model/vlm.py",
        "vlm_model/language_model.py",
        "vlm_model/vision_encoder.py",
        "vlm_model/connector.py",
        "vlm_model/utils.py",
    ),
    "qwen3_vl": (),
    "astrollava": (),
}


class ProtocolError(ValueError):
    """Raised when a paper protocol is incomplete or internally inconsistent."""


def _canonical(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _canonical(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    return value


def canonical_json(value: Any) -> str:
    """Return the stable serialization used for every protocol fingerprint."""

    return json.dumps(
        _canonical(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _require(mapping: Mapping[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise ProtocolError(f"Missing {context}.{key}")
    return mapping[key]


def _require_sha(value: Any, context: str, length: int = 40) -> None:
    text = str(value or "").lower()
    pattern = SHA_RE if length == 40 else SHA256_RE
    if not pattern.fullmatch(text):
        raise ProtocolError(f"{context} must be a {length}-character lowercase hex digest")


def validate_protocol(data: Mapping[str, Any]) -> None:
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ProtocolError(
            f"schema_version must be {SCHEMA_VERSION}, got {data.get('schema_version')!r}"
        )

    study = _require(data, "study", "protocol")
    for flag in ("retrieval", "few_shot", "quantization"):
        if study.get(flag) is not False:
            raise ProtocolError(f"study.{flag} must be explicitly false for the frozen protocol")

    runtime = _require(data, "runtime", "protocol")
    if not re.fullmatch(r"\d+\.\d+\.\d+", str(runtime.get("bootstrap_uv_version") or "")):
        raise ProtocolError("runtime.bootstrap_uv_version must pin major.minor.patch")
    if int(runtime.get("minimum_gpu_memory_mib") or 0) < 1:
        raise ProtocolError("runtime.minimum_gpu_memory_mib must be positive")
    minimum_capability = float(runtime.get("minimum_compute_capability") or 0)
    maximum_capability = float(runtime.get("maximum_compute_capability_exclusive") or 0)
    if minimum_capability < 1 or maximum_capability <= minimum_capability:
        raise ProtocolError("runtime compute-capability bounds are invalid")

    datasets = _require(data, "datasets", "protocol")
    generation = _require(data, "generation", "protocol")
    models = _require(data, "models", "protocol")
    environments = _require(data, "environments", "protocol")
    for name, environment in environments.items():
        if not re.fullmatch(r"\d+\.\d+\.\d+", str(environment.get("python") or "")):
            raise ProtocolError(f"environments.{name}.python must pin major.minor.patch")
        if not str(environment.get("torch_index_url") or "").startswith("https://"):
            raise ProtocolError(f"environments.{name}.torch_index_url must be pinned")
        packages = environment.get("packages")
        if not isinstance(packages, Mapping) or not packages:
            raise ProtocolError(f"environments.{name}.packages must be a non-empty mapping")
        if any(not str(version).strip() for version in packages.values()):
            raise ProtocolError(f"environments.{name}.packages contains an empty version")
    for suite in SUPPORTED_SUITES:
        _require(datasets, suite, "datasets")
        _require(generation, suite, "generation")
        if datasets[suite].get("rag_used") is not False:
            raise ProtocolError(f"datasets.{suite}.rag_used must be explicitly false")

    if datasets["internal"].get("expected_records") != 3271:
        raise ProtocolError("The frozen internal protocol must contain exactly 3,271 records")
    _require_sha(datasets["internal"].get("source_revision"), "datasets.internal.source_revision")
    _require_sha(
        datasets["internal"].get("test_json_sha256"),
        "datasets.internal.test_json_sha256",
        64,
    )
    _require_sha(datasets["deepsdo"].get("archive_sha256"), "datasets.deepsdo.archive_sha256", 64)

    for label, model in models.items():
        suites = model.get("suites") or []
        if not suites or any(suite not in SUPPORTED_SUITES for suite in suites):
            raise ProtocolError(f"models.{label}.suites contains an unsupported or empty suite")
        _require_sha(model.get("revision"), f"models.{label}.revision")
        environment_name = model.get("environment")
        if environment_name not in environments:
            raise ProtocolError(f"models.{label}.environment is missing or unknown")
        if model.get("backend") == "astrollava":
            _require_sha(model.get("code_revision"), f"models.{label}.code_revision")
            vision = model.get("vision_encoder") or {}
            if not vision.get("repo_id"):
                raise ProtocolError(f"models.{label}.vision_encoder.repo_id is required")
            _require_sha(
                vision.get("revision"), f"models.{label}.vision_encoder.revision"
            )
            if model.get("load_4bit") or model.get("load_8bit") or model.get("flash_attention"):
                raise ProtocolError("AstroLLaVA paper runs must be unquantized without FlashAttention")
        for digest_key in (
            "checkpoint_sha256",
            "connector_sha256",
            "lora_sha256",
        ):
            if digest_key in model:
                _require_sha(model[digest_key], f"models.{label}.{digest_key}", 64)

    base_models = _require(data, "base_models", "protocol")
    for key, item in base_models.items():
        _require_sha(item.get("revision"), f"base_models.{key}.revision")
    for key, item in _require(data, "scorers", "protocol").items():
        _require_sha(item.get("revision"), f"scorers.{key}.revision")

    common = generation.get("common") or {}
    if common.get("do_sample") is not False or float(common.get("temperature", -1)) != 0.0:
        raise ProtocolError("Paper generation must use deterministic greedy decoding")
    if common.get("num_beams") != 1:
        raise ProtocolError("Paper generation must use exactly one beam")


@dataclass(frozen=True)
class PaperProtocol:
    path: Path
    data: Dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "PaperProtocol":
        protocol_path = Path(path).resolve()
        with protocol_path.open("r", encoding="utf-8") as stream:
            loaded = yaml.safe_load(stream)
        if not isinstance(loaded, dict):
            raise ProtocolError(f"Protocol must be a YAML mapping: {protocol_path}")
        validate_protocol(loaded)
        return cls(protocol_path, loaded)

    @property
    def study_id(self) -> str:
        return str(self.data["study"]["id"])

    @property
    def fingerprint(self) -> str:
        return sha256_json(self.data)

    def selected_models(self, suite: str) -> Dict[str, Dict[str, Any]]:
        self._check_suite(suite)
        return {
            label: copy.deepcopy(model)
            for label, model in self.data["models"].items()
            if suite in model.get("suites", [])
        }

    def generation_suite_payload(self, suite: str) -> Dict[str, Any]:
        """Return config values that can change GPU generation for one suite."""

        self._check_suite(suite)
        study = self.data["study"]
        return {
            "schema_version": self.data["schema_version"],
            "study": {
                key: study[key]
                for key in ("id", "seed", "retrieval", "few_shot", "quantization", "prompt_ablation")
            },
            "dataset": self.data["datasets"][suite],
            "generation_common": self.data["generation"]["common"],
            "generation_suite": self.data["generation"][suite],
            "models": self.selected_models(suite),
            "base_models": self.data["base_models"],
        }

    def analysis_suite_payload(self, suite: str) -> Dict[str, Any]:
        """Return analysis/report values without moving completed generation runs."""

        self._check_suite(suite)
        statistics = self.data["statistics"]
        suite_statistics: Dict[str, Any] = {
            "bootstrap_replicates": statistics["bootstrap_replicates"],
            "seed": statistics["seed"],
            "confidence_level": statistics["confidence_level"],
            "predeclared_comparisons": statistics["predeclared_comparisons"][suite],
        }
        if suite == "internal":
            suite_statistics["cluster_key"] = statistics["internal_cluster_key"]
        elif suite == "deepsdo":
            suite_statistics["primary_metric"] = statistics["deepsdo_primary_metric"]

        if suite == "internal":
            scorer_names = ("sbert", "nli")
            package_names = ("pycocoevalcap", "sentence-transformers", "matplotlib")
        elif suite == "deepsdo":
            scorer_names = ()
            package_names = ("pycocoevalcap", "matplotlib")
        else:
            scorer_names = ()
            package_names = ("matplotlib",)
        return {
            "schema_version": self.data["schema_version"],
            "generation_suite_sha256": self.suite_fingerprint(suite),
            "scorers": {
                name: self.data["scorers"][name]
                for name in scorer_names
            },
            "metric_packages": {
                name: self.data.get("metric_packages", {})[name]
                for name in package_names
            },
            "statistics": suite_statistics,
            "reporting": self.data["reporting"],
        }

    # Backward-compatible name used by existing callers/tests. Its boundary is
    # deliberately generation-only so offline scoring changes never force GPU
    # inference to be repeated.
    def suite_payload(self, suite: str) -> Dict[str, Any]:
        return self.generation_suite_payload(suite)

    def suite_fingerprint(self, suite: str) -> str:
        return sha256_json(self.generation_suite_payload(suite))

    def analysis_fingerprint(self, suite: str) -> str:
        return sha256_json(self.analysis_suite_payload(suite))

    def model_payload(self, suite: str, model_label: str) -> Dict[str, Any]:
        models = self.selected_models(suite)
        if model_label not in models:
            raise ProtocolError(f"Model {model_label!r} is not enabled for suite {suite!r}")
        payload = self.generation_suite_payload(suite)
        payload["models"] = {model_label: models[model_label]}
        environment_name = models[model_label]["environment"]
        payload["environment"] = {
            environment_name: copy.deepcopy(self.data["environments"][environment_name])
        }
        return payload

    def model_fingerprint(self, suite: str, model_label: str) -> str:
        return sha256_json(self.model_payload(suite, model_label))

    def generation_implementation_payload(
        self, model_label: str, repo_root: str | Path
    ) -> Dict[str, Any]:
        model = self.data["models"].get(model_label)
        if model is None:
            raise ProtocolError(f"Unknown model {model_label!r}")
        root = Path(repo_root).resolve()
        relative_paths = tuple(COMMON_GENERATION_FILES) + tuple(
            BACKEND_GENERATION_FILES.get(str(model["backend"]), ())
        )
        files: Dict[str, str] = {}
        for relative in sorted(set(relative_paths)):
            path = root / relative
            if not path.is_file():
                raise ProtocolError(f"Generation implementation file is missing: {path}")
            files[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        symbols: Dict[str, str] = {}
        for relative, required_names in GENERATION_SYMBOLS.items():
            path = root / relative
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
            found: Dict[str, str] = {}
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    if node.name in required_names:
                        segment = ast.get_source_segment(source, node)
                        if segment is not None:
                            found[node.name] = segment
            missing = sorted(set(required_names) - set(found))
            if missing:
                raise ProtocolError(
                    f"Generation implementation symbols missing from {relative}: {missing}"
                )
            symbols[relative] = sha256_json(found)
        return {
            "backend": model["backend"],
            "files": files,
            "selected_symbols": symbols,
            "external_code_revision": model.get("code_revision"),
        }

    def effective_model_fingerprint(
        self,
        suite: str,
        model_label: str,
        records_sha256: str,
        repo_root: str | Path,
    ) -> str:
        _require_sha(records_sha256, "records_sha256", 64)
        return sha256_json(
            {
                "model_generation_config_sha256": self.model_fingerprint(suite, model_label),
                "records_sha256": records_sha256,
                "implementation": self.generation_implementation_payload(model_label, repo_root),
            }
        )

    def model_output_dir(
        self,
        suite: str,
        model_label: str,
        records_sha256: str,
        repo_root: str | Path,
        root: str | Path | None = None,
    ) -> Path:
        effective = self.effective_model_fingerprint(
            suite, model_label, records_sha256, repo_root
        )
        return self.output_dir(suite, root) / model_label / effective[:16]

    def output_dir(self, suite: str, root: str | Path | None = None) -> Path:
        base = Path(root or self.data["runtime"]["output_root"])
        return base / suite / self.suite_fingerprint(suite)[:16]

    def astraq_architecture(self, model_label: str) -> Dict[str, Any]:
        model = self.data["models"].get(model_label)
        if not model or model.get("backend") != "astraq":
            raise ProtocolError(f"{model_label!r} is not an AstraQ backend")
        config: Dict[str, Any] = {
            "vision_encoder": {
                "model_name": self.data["base_models"]["vision_encoder"]["repo_id"],
                "revision": self.data["base_models"]["vision_encoder"]["revision"],
                "select_layer": -2,
                "select_feature": "patch",
            },
            "language_model": {
                "model_name": self.data["base_models"]["language_model"]["repo_id"],
                "revision": self.data["base_models"]["language_model"]["revision"],
                "torch_dtype": model["dtype"],
            },
            "connector": {"vision_hidden_size": 1024, "llm_hidden_size": 1536},
        }
        if model.get("stage") == 2:
            config["language_model"]["lora"] = copy.deepcopy(model["lora"])
        return config

    def _check_suite(self, suite: str) -> None:
        if suite not in SUPPORTED_SUITES:
            raise ProtocolError(f"Unknown suite {suite!r}; expected one of {SUPPORTED_SUITES}")


def parse_csv_selection(value: str | Sequence[str], allowed: Iterable[str]) -> list[str]:
    if isinstance(value, str):
        selected = [item.strip() for item in value.split(",") if item.strip()]
    else:
        selected = [str(item).strip() for item in value if str(item).strip()]
    allowed_set = set(allowed)
    if selected == ["all"]:
        return sorted(allowed_set)
    unknown = sorted(set(selected) - allowed_set)
    if unknown:
        raise ProtocolError(f"Unknown selection(s): {', '.join(unknown)}")
    if not selected:
        raise ProtocolError("At least one item must be selected")
    return selected
