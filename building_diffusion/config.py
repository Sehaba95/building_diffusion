"""
Typed configurations for the training and evaluation scripts.

Each script reads one YAML file (configs/*.yaml) and accepts dotted overrides on the command line:
    python scripts/train_diffusion.py model.layer_name=GAT train.epochs=1
Every key must be set in the YAML file: unknown, missing or wrongly typed keys raise an error.
"""
import argparse
from dataclasses import asdict, dataclass, fields, is_dataclass
from typing import Optional, Union, get_args, get_origin

import yaml


@dataclass
class DataConfig:
    """data: dataset folders and split sizes."""
    processed_dir: str  # Processed graphs (.pt), written by scripts/preprocess.py
    raw_dir: str  # Raw dataset (global_graph_data/, local_graph_data/, voxel_data/)
    train_size: int  # Train + validation buildings (split 80/20)
    test_size: int


@dataclass
class TrainConfig:
    """train: optimization settings."""
    lr: float
    epochs: int
    batch_size: int
    seed: int  # Random seed, set before the data loaders and the model are created
    deterministic: bool  # Deterministic GPU algorithms: identical GPU runs, but slower and much more GPU memory


@dataclass
class SSLModelConfig:
    """model: architecture of the SSL program graph encoder."""
    layer_name: str  # GAT, GATv2, GCN, GatedGraphConv, TransformerConv, GIN, SAGE
    in_channels: int
    encoding: int  # Pre-processing size
    hidden_channels: list[int]  # The last value is the graph embedding size
    out_channels: int  # Post-processing size
    heads: int
    concat: bool  # Concatenate (True) or average (False) the attention heads
    dropout: float


@dataclass
class DiffusionModelConfig:
    """model: architecture of the Building-Diffusion model."""
    layer_name: str  # GAT, GATv2, GCN, GatedGraphConv, TransformerConv, GIN, SAGE
    in_channels: int
    encoding: int  # Pre-processing size
    hidden_channels: list[int]  # The first value is also the SSL embedding size
    out_channels: int  # Post-processing size
    heads: int
    concat: bool
    residual: bool
    floor_pos_enc: bool  # Add the floor positional encoding
    dropout: float
    timestep_embed_size: int
    diffusion_steps: int
    ssl_checkpoint: Optional[str]  # Trained SSL model; null: default checkpoint for hidden_channels[0]


@dataclass
class EvalSettings:
    """eval: evaluation settings."""
    num_samples: int  # Samples generated per building; the one with the lowest FGW distance is kept
    seed: int
    deterministic: bool  # Deterministic GPU algorithms (see train.deterministic)
    split_batch_size: int  # Test size is rounded to a multiple of it (32: batch size used in training)
    layer_name: Optional[str]  # Only for old runs whose log does not contain the layer name


@dataclass
class SSLConfig:
    """configs/ssl.yaml"""
    data: DataConfig
    model: SSLModelConfig
    train: TrainConfig


@dataclass
class DiffusionConfig:
    """configs/diffusion.yaml"""
    data: DataConfig
    model: DiffusionModelConfig
    train: TrainConfig


@dataclass
class EvaluateConfig:
    """configs/evaluate.yaml (the model settings come from the evaluated run)"""
    data: DataConfig
    eval: EvalSettings


def _coerce(value, field_type, key):
    """Check value against field_type; ints are accepted for floats (YAML reads 1e-3 as a string, so it is converted too)."""
    if get_origin(field_type) is Union:  # Optional[X]
        if value is None:
            return None
        field_type = next(arg for arg in get_args(field_type) if arg is not type(None))

    if field_type is float and isinstance(value, (int, float, str)) and not isinstance(value, bool):
        try:
            return float(value)
        except ValueError:
            pass
    elif get_origin(field_type) is list and isinstance(value, list):
        item_type = get_args(field_type)[0]
        return [_coerce(item, item_type, f"{key}[{i}]") for i, item in enumerate(value)]
    elif isinstance(value, field_type) and not (field_type is int and isinstance(value, bool)):
        return value

    raise TypeError(f"Config key {key!r}: expected {field_type.__name__}, got {value!r}")


def build_config(cls, values, prefix=""):
    """Build the dataclass cls from a dict, checking keys and value types."""
    if not isinstance(values, dict):
        raise TypeError(f"Config section {prefix.rstrip('.')!r} must be a mapping")
    names = [f.name for f in fields(cls)]
    unknown = sorted(set(values) - set(names))
    missing = [name for name in names if name not in values]
    if unknown or missing:
        raise KeyError(f"Config {prefix or 'root'}: unknown keys {unknown}, missing keys {missing}")

    kwargs = {}
    for f in fields(cls):
        key = prefix + f.name
        if is_dataclass(f.type):
            kwargs[f.name] = build_config(f.type, values[f.name], key + ".")
        else:
            kwargs[f.name] = _coerce(values[f.name], f.type, key)
    return cls(**kwargs)


def load_config(cls, path, overrides=()):
    """Read a YAML file into the dataclass cls, after applying "section.key=value" overrides."""
    with open(path, "r") as file:
        values = yaml.safe_load(file)

    for override in overrides:
        key, sep, value = override.partition("=")
        if not sep:
            raise ValueError(f"Override {override!r} must look like section.key=value")
        node = values
        *sections, leaf = key.split(".")
        for section in sections:
            if not isinstance(node, dict) or section not in node:
                raise KeyError(f"Unknown config key: {key}")
            node = node[section]
        if not isinstance(node, dict) or leaf not in node:
            raise KeyError(f"Unknown config key: {key}")
        parsed = yaml.safe_load(value)
        # YAML 1.1 also reads yes/no/on/off as booleans: only true/false are booleans here
        if isinstance(parsed, bool) and value.strip().lower() not in ("true", "false"):
            parsed = value
        node[leaf] = parsed

    return build_config(cls, values)


def save_config(config, path):
    """Write a config dataclass to a YAML file (the format read by load_config)."""
    with open(path, "w") as file:
        yaml.safe_dump(asdict(config), file, sort_keys=False)


def config_parser(description, default_config):
    """Argument parser with --config and the positional dotted overrides."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=default_config, help=f"YAML config file (default: {default_config})")
    parser.add_argument("overrides", nargs="*", help="Config overrides, e.g. train.epochs=1 model.layer_name=GAT")
    return parser
