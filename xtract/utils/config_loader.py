import os
import yaml


def load_yaml(path):
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return _resolve_env_vars(config)


def load_config(path="config/settings.yaml"):
    return load_yaml(path)


def _resolve_env_vars(config):
    if isinstance(config, dict):
        return {k: _resolve_env_vars(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [_resolve_env_vars(item) for item in config]
    elif isinstance(config, str) and config.startswith("${") and config.endswith("}"):
        env_var = config[2:-1]
        return os.getenv(env_var, config)
    return config
