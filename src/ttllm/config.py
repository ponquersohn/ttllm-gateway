"""Config Module.

Settings are resolved in order: YAML config file -> environment variables -> defaults.
When a YAML config file is provided via TTLLM_CONFIG_FILE (and optionally TTLLM_CONFIG_ENV),
the ConfigLoader loads and resolves it (env://, secret://, includes, refs), then the
resulting dict is fed into the pydantic Settings model for typed validation.
"""

import json
import logging
import os
from functools import wraps
from pathlib import Path
from typing import Union

import boto3
import yaml
from lru import LRU
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseConfig(BaseModel):
    url: str = "postgresql+asyncpg://ttllm:dev@localhost:5432/ttllm"
    pool_size: int = 5


class EngineConfig(BaseModel):
    base_url: str = "http://localhost:4000"
    cors_origins: list[str] = ["*"]
    log_request_bodies: bool = False


class JWTConfig(BaseModel):
    secret_key: str = "CHANGE-ME-IN-PRODUCTION"
    algorithm: str = "HS256"
    access_token_ttl_minutes: int = 15
    refresh_token_ttl_days: int = 30
    token_ttl_days: int = 30
    token_max_ttl_days: int = 365


class IdPConfig(BaseModel):
    name: str
    type: str = "oidc"
    client_id: str = ""
    client_secret: str = ""
    tenant_id: str = ""
    discovery_url: str | None = None
    scopes: list[str] = ["openid", "profile", "email"]
    group_mapping: dict[str, list[str]] = {}
    default_groups: list[str] = []

    def get_discovery_url(self) -> str:
        if self.discovery_url:
            return self.discovery_url
        if self.tenant_id:
            return f"https://login.microsoftonline.com/{self.tenant_id}/v2.0/.well-known/openid-configuration"
        raise ValueError(f"IdP '{self.name}': either discovery_url or tenant_id must be set")


class AuthConfig(BaseModel):
    jwt: JWTConfig = JWTConfig()
    identity_providers: dict[str, IdPConfig] = {}


class ProviderConfig(BaseModel):
    default_region: str = "us-east-1"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TTLLM_", env_nested_delimiter="__")

    database: DatabaseConfig = DatabaseConfig()
    engine: EngineConfig = EngineConfig()
    auth: AuthConfig = AuthConfig()
    provider: ProviderConfig = ProviderConfig()


# --- YAML ConfigLoader ---


def merge_dicts(source: dict, destination: dict):
    """Merges two dicts, overwriting destination values with source.

    WARNING: Modifies and returns `destination`.
    """
    for key, value in source.items():
        if isinstance(value, dict):
            node = destination.setdefault(key, {})
            merge_dicts(value, node)
        else:
            destination[key] = value
    return destination


def resolve_value(value, logger: logging.Logger):
    """Resolve a single value with env:// or secret:// prefix.

    Supported patterns:
    - env://ENV_VAR_NAME,default_value - Environment variable with default
    - env://ENV_VAR_NAME - Environment variable or None
    - secret://arn - AWS Secrets Manager secret value
    """
    if not isinstance(value, str):
        return value

    if value.startswith("env://"):
        rest = value[6:]
        if "," in rest:
            env_name, default = rest.split(",", 1)
        else:
            env_name = rest
            default = None

        result = os.getenv(env_name, default)
        logger.debug(f"Resolved env://{env_name} to {'<set>' if result else '<empty>'}")
        return result

    elif value.startswith("secret://"):
        arn = value[9:]
        try:
            session = boto3.session.Session()
            client = session.client(service_name="secretsmanager")
            logger.debug(f"Fetching secret from AWS Secrets Manager: {arn}")
            resp = client.get_secret_value(SecretId=arn)

            if "SecretString" in resp:
                secret = resp["SecretString"]
                try:
                    return json.loads(secret)
                except json.JSONDecodeError:
                    return secret
            else:
                return resp["SecretBinary"]
        except Exception as e:
            logger.error(f"Failed to resolve secret://{arn}: {e}")
            raise

    return value


def resolve_dict(config, logger: logging.Logger):
    """Recursively resolve all values in a dictionary."""
    if not isinstance(config, dict):
        return config

    resolved = {}
    for key, value in config.items():
        if isinstance(value, dict):
            resolved[key] = resolve_dict(value, logger)
        elif isinstance(value, list):
            resolved[key] = [
                (resolve_dict(item, logger) if isinstance(item, dict) else resolve_value(item, logger))
                for item in value
            ]
        else:
            resolved[key] = resolve_value(value, logger)
    return resolved


def resolve(func):
    """Decorator to resolve env:// and secret:// patterns in config."""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        config = func(self, *args, **kwargs)
        return resolve_dict(config, self.logger)

    return wrapper


class ConfigEmptyException(Exception):
    """Raised when a config file is empty."""


class ConfigEnvironmentNotFoundException(Exception):
    """Raised when the requested environment section is not in the config."""


class ConfigNameNotFound(Exception):
    """Raised when config name was passed but not found."""


class ConfigFileNotFound(Exception):
    """Raised when the config file does not exist."""


class ConfigIncludeError(Exception):
    """Raised when an include directive fails."""


class ConfigRefError(Exception):
    """Raised when a ref directive fails."""


class ConfigLoader:
    """YAML config loader with environment sections, includes, refs, and caching.

    Features:
    - env:// and secret:// value resolution
    - YAML file includes with circular dependency detection
    - ref: cross-references via dotted paths
    - Local config overrides (local.<filename>)
    - Config inheritance from parent directories
    - LRU caching
    """

    _CACHE_SIZE = 100
    _cache: LRU = LRU(_CACHE_SIZE)

    @staticmethod
    def _cache_key(config_file: str, environment: str, load_local: bool):
        return f"{config_file}|{environment}|{load_local}"

    @classmethod
    def clear_cache(cls):
        cls._cache.clear()

    def _process_includes(self, config, base_dir, include_stack=None):
        if include_stack is None:
            include_stack = set()

        if isinstance(config, list):
            return [self._process_includes(item, base_dir, include_stack) for item in config]

        if not isinstance(config, dict):
            return config

        if "include" not in config:
            return {k: self._process_includes(v, base_dir, include_stack) for k, v in config.items()}

        include_val = config["include"]
        if isinstance(include_val, str):
            include_paths = [include_val]
        elif isinstance(include_val, list):
            include_paths = include_val
        else:
            raise ConfigIncludeError(f"Invalid include value: {include_val!r} (must be string or list)")

        merged = {}
        for inc_path in include_paths:
            resolved = (base_dir / inc_path).resolve()
            if not resolved.is_file():
                raise ConfigIncludeError(f"Include file not found: {resolved}")
            if str(resolved) in include_stack:
                raise ConfigIncludeError(f"Circular include detected: {resolved}")

            new_stack = include_stack | {str(resolved)}
            with resolved.open(encoding="utf-8", mode="r") as f:
                inc_config = yaml.load(f, Loader=yaml.FullLoader)

            if inc_config and isinstance(inc_config, dict):
                inc_config = self._process_includes(inc_config, resolved.parent, new_stack)
            elif inc_config is None:
                inc_config = {}

            merge_dicts(inc_config, merged)

        overrides = {
            k: self._process_includes(v, base_dir, include_stack) for k, v in config.items() if k != "include"
        }
        if overrides:
            merge_dicts(overrides, merged)

        return merged

    def _resolve_refs(self, config, _root=None, _resolving=None):
        if _root is None:
            _root = config
        if _resolving is None:
            _resolving = set()

        if isinstance(config, list):
            return [self._resolve_refs(item, _root, _resolving) for item in config]

        if isinstance(config, dict):
            return {k: self._resolve_refs(v, _root, _resolving) for k, v in config.items()}

        if isinstance(config, str) and config.startswith("ref:"):
            ref_path = config[4:]
            if ref_path in _resolving:
                raise ConfigRefError(f"Circular ref detected: {ref_path}")

            parts = ref_path.split(".")
            current = _root
            for part in parts:
                if not isinstance(current, dict) or part not in current:
                    raise ConfigRefError(f"Ref path not found: {ref_path}")
                current = current[part]

            return self._resolve_refs(current, _root, _resolving | {ref_path})

        return config

    def _load_config_file(self, config_file_path: Path, load_local: bool = True):
        with config_file_path.open(encoding="utf-8", mode="r") as config_file:
            self.logger.debug(f"Attempting to load config file {config_file_path}")
            config = yaml.load(config_file, Loader=yaml.FullLoader)

            if config and isinstance(config, dict):
                config = self._process_includes(config, config_file_path.parent)

            if load_local:
                local_config_file_path = config_file_path.parent / Path("local." + config_file_path.name)

                if local_config_file_path.is_file():
                    try:
                        local_loaded_config_file = self._load_config_file(local_config_file_path, load_local=False)
                        config = merge_dicts(local_loaded_config_file, config)
                    except (
                        FileNotFoundError,
                        IOError,
                        ConfigEmptyException,
                        ConfigEnvironmentNotFoundException,
                    ) as e:
                        self.logger.warning(
                            f"Local config {local_config_file_path} exists but unable to load: {repr(e)} - ignoring"
                        )

            if not config:
                raise ConfigEmptyException()

            if config.get("inherit", False):
                parent_config_file_path = config_file_path.parent.parent / "config.yaml"

                if parent_config_file_path.is_file():
                    self.logger.debug(f"Loading parent settings from {parent_config_file_path}")
                    parent_loaded_config_file = self._load_config_file(parent_config_file_path)
                    config = merge_dicts(config, parent_loaded_config_file)

            return config

    def __init__(
        self,
        config_file: Union[Path, str] = None,
        environment: str = "dev",
        load_local: bool = True,
        logger: logging.Logger = None,
    ):
        super().__init__()
        self.logger = logger or logging.getLogger("root")
        self.environment = environment
        self.load_local = load_local
        self.config_file = config_file

        if config_file is not None:
            self.config = self.load_config(config_file)

    @resolve
    def load_config(
        self,
        config_file: Union[Path, str] = None,
        environment: str = None,
        load_local: bool = None,
    ):
        if not environment:
            environment = self.environment
        if not config_file:
            config_file = self.config_file
        if load_local is None:
            load_local = self.load_local

        key = ConfigLoader._cache_key(config_file, environment, load_local)
        if key in ConfigLoader._cache:
            return ConfigLoader._cache[key]

        self.logger.debug(f"Loading settings from {config_file} for environment {environment}")

        config_file_path = config_file if isinstance(config_file, Path) else Path(config_file)

        if not config_file_path.is_file():
            raise ConfigFileNotFound(f"Config file: {config_file_path} not found.")

        loaded_config_file = self._load_config_file(config_file_path)
        config = None

        if not environment:
            config = loaded_config_file
        else:
            if environment not in loaded_config_file:
                raise ConfigEnvironmentNotFoundException(f"Environment {environment} not found in config file")

            default_config = loaded_config_file.get("default", {})
            if default_config:
                loaded_config_file[environment] = merge_dicts(loaded_config_file[environment], default_config)

            config = loaded_config_file[environment]

        if environment and isinstance(config, dict):
            config = self._resolve_refs(config)

        ConfigLoader._cache[key] = config
        return config


def load_settings(
    config_file: Union[Path, str] = None,
    environment: str = None,
) -> Settings:
    """Build Settings from a YAML config file, falling back to env vars + defaults.

    If config_file is not provided, checks TTLLM_CONFIG_FILE env var.
    If environment is not provided, checks TTLLM_CONFIG_ENV env var (default: "dev").
    If no config file is found, returns plain Settings (env vars + defaults only).
    """
    config_file = config_file or os.getenv("TTLLM_CONFIG_FILE")
    if config_file is None:
        return Settings()

    environment = environment or os.getenv("TTLLM_CONFIG_ENV", "dev")
    config_dict = ConfigLoader(
        config_file=config_file,
        environment=environment,
    ).config

    return Settings(**config_dict)


settings = load_settings()
