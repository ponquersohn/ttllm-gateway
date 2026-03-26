"""Tests for ConfigLoader include, ref, and resolution features."""

import logging
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from ttllm.config import (
    ConfigEmptyException,
    ConfigEnvironmentNotFoundException,
    ConfigFileNotFound,
    ConfigIncludeError,
    ConfigLoader,
    ConfigRefError,
    merge_dicts,
    resolve_dict,
)

FIXTURES_DIR = Path(__file__).parent / "config_fixtures"


@pytest.fixture(autouse=True)
def clear_config_cache():
    """Clear ConfigLoader cache before each test."""
    ConfigLoader.clear_cache()
    yield
    ConfigLoader.clear_cache()


def make_loader(environment="dev"):
    return ConfigLoader(logger=logging.getLogger("test"), environment=environment)


class TestIncludeDirective:
    """Tests for the include directive in config files."""

    def test_single_include(self):
        loader = make_loader()
        config = loader._load_config_file(FIXTURES_DIR / "with_single_include.yaml", load_local=False)
        assert config["default"]["settings"]["sqs"]["region"] == "us-east-1"
        assert config["default"]["settings"]["sqs"]["visibility_timeout"] == 300
        assert config["default"]["settings"]["sqs"]["wait_time_seconds"] == 20
        assert config["default"]["settings"]["sqs"]["max_messages"] == 1

    def test_multi_include(self):
        loader = make_loader()
        config = loader._load_config_file(FIXTURES_DIR / "with_multi_include.yaml", load_local=False)
        default_settings = config["default"]["settings"]
        assert default_settings["sqs"]["region"] == "us-east-1"
        assert default_settings["sqs"]["wait_time_seconds"] == 20
        assert default_settings["outputs"]["main"]["type"] == "s3"
        assert default_settings["outputs"]["main"]["params"]["bucket"] == "shared-bucket"

    def test_include_with_overrides(self):
        loader = make_loader()
        config = loader._load_config_file(FIXTURES_DIR / "with_include_overrides.yaml", load_local=False)
        sqs = config["default"]["settings"]["sqs"]
        assert sqs["visibility_timeout"] == 600
        assert sqs["queue_name"] == "overridden-queue"
        assert sqs["wait_time_seconds"] == 20
        assert sqs["max_messages"] == 1

    def test_root_level_include(self):
        loader = make_loader()
        config = loader._load_config_file(FIXTURES_DIR / "with_root_include.yaml", load_local=False)
        assert config["sqs"]["region"] == "us-east-1"
        assert config["sqs"]["visibility_timeout"] == 300
        assert config["extra"]["enabled"] is True

    def test_nested_include_chain(self):
        loader = make_loader()
        config = loader._load_config_file(FIXTURES_DIR / "with_nested_include.yaml", load_local=False)
        data = config["default"]["data"]
        assert data["deep_value"] == "hello_from_deep"
        assert data["nested_data"]["key1"] == "value1"
        assert data["nested_data"]["key2"] == "value2"
        assert data["extra_key"] == "from_nested_include"

    def test_relative_paths(self):
        """Include paths are relative to the file containing the include."""
        loader = make_loader()
        config = loader._load_config_file(FIXTURES_DIR / "with_nested_include.yaml", load_local=False)
        assert config["default"]["data"]["deep_value"] == "hello_from_deep"

    def test_nested_include_with_sibling_overrides(self):
        """Include inside a nested dict merges with sibling keys and deep-merges params."""
        loader = make_loader()
        config = loader._load_config_file(FIXTURES_DIR / "with_nested_include_override.yaml", load_local=False)
        splunk = config["default"]["outputs"]["splunk"]
        assert splunk["type"] == "splunk_hec"
        assert splunk["params"]["hec_url"] == "https://splunk.example.com"
        assert splunk["params"]["batch_size"] == 100
        assert splunk["params"]["verify_ssl"] is True
        assert splunk["params"]["timeout"] == 60
        assert splunk["checkpoint"] is True
        assert splunk["params"]["index"] == "my_index"
        assert splunk["params"]["sourcetype"] == "test:events"

    def test_nested_include_with_env_merge(self):
        """Environment overrides merge on top of default (which already resolved the include)."""
        loader = make_loader("dev")
        config = loader.load_config(FIXTURES_DIR / "with_nested_include_override.yaml")
        splunk = config["outputs"]["splunk"]
        assert splunk["type"] == "splunk_hec"
        assert splunk["params"]["hec_url"] == "https://splunk.example.com"
        assert splunk["params"]["verify_ssl"] is True
        assert splunk["params"]["timeout"] == 60
        assert splunk["checkpoint"] is True
        assert splunk["params"]["sourcetype"] == "test:events"
        assert splunk["params"]["batch_size"] == 50
        assert splunk["params"]["index"] == "dev_index"

    def test_missing_include_file_raises_error(self):
        loader = make_loader()
        with pytest.raises(ConfigIncludeError, match="Include file not found"):
            loader._load_config_file(FIXTURES_DIR / "with_missing_include.yaml", load_local=False)

    def test_circular_include_raises_error(self):
        loader = make_loader()
        with pytest.raises(ConfigIncludeError, match="Circular include detected"):
            loader._load_config_file(FIXTURES_DIR / "with_circular_include_a.yaml", load_local=False)


class TestRefDirective:
    """Tests for the ref: directive in config values."""

    def test_basic_ref_resolution(self):
        loader = make_loader("dev")
        config = loader.load_config(FIXTURES_DIR / "with_refs.yaml")
        assert config["sqs"]["region"] == "us-east-1"

    def test_env_specific_ref_resolution_dev(self):
        loader = make_loader("dev")
        config = loader.load_config(FIXTURES_DIR / "with_refs.yaml")
        assert config["outputs"]["main"]["params"]["bucket"] == "my-app-dev"

    def test_env_specific_ref_resolution_prod(self):
        loader = make_loader("prod")
        config = loader.load_config(FIXTURES_DIR / "with_refs.yaml")
        assert config["outputs"]["main"]["params"]["bucket"] == "my-app-prod"

    def test_nested_ref_path(self):
        loader = make_loader("dev")
        config = loader.load_config(FIXTURES_DIR / "with_nested_refs.yaml")
        assert config["redis"]["host"] == "dev-redis.example.com"
        assert config["redis"]["port"] == 6379

    def test_dict_ref(self):
        loader = make_loader("dev")
        config = loader.load_config(FIXTURES_DIR / "with_dict_ref.yaml")
        assert config["outputs"]["main"] == {"type": "s3", "params": {"output_compression": "gz"}}

    def test_missing_ref_raises_error(self):
        loader = make_loader("dev")
        with pytest.raises(ConfigRefError, match="Ref path not found"):
            loader.load_config(FIXTURES_DIR / "with_missing_ref.yaml")

    def test_circular_ref_raises_error(self):
        loader = make_loader("dev")
        with pytest.raises(ConfigRefError, match="Circular ref detected"):
            loader.load_config(FIXTURES_DIR / "with_circular_ref.yaml")

    def test_chained_refs(self):
        """A ref target can itself be a ref (resolved via recursion)."""
        loader = make_loader()
        config = {
            "vars": {"region": "us-east-1"},
            "alias": "ref:vars.region",
            "double_alias": "ref:alias",
        }
        resolved = loader._resolve_refs(config)
        assert resolved["alias"] == "us-east-1"
        assert resolved["double_alias"] == "us-east-1"

    def test_ref_in_list(self):
        """Refs inside list values are resolved."""
        loader = make_loader()
        config = {
            "vars": {"region": "us-east-1"},
            "regions": ["ref:vars.region", "us-west-2"],
        }
        resolved = loader._resolve_refs(config)
        assert resolved["regions"] == ["us-east-1", "us-west-2"]

    def test_non_ref_strings_unchanged(self):
        """Strings that don't start with ref: are left unchanged."""
        loader = make_loader()
        config = {
            "name": "my-app",
            "env_val": "env://MY_VAR,default",
            "number": 42,
        }
        resolved = loader._resolve_refs(config)
        assert resolved["name"] == "my-app"
        assert resolved["env_val"] == "env://MY_VAR,default"
        assert resolved["number"] == 42


class TestIncludeAndRefCombined:
    """Tests for using include and ref together."""

    def test_ref_to_included_content(self):
        loader = make_loader("dev")
        config = loader.load_config(FIXTURES_DIR / "with_include_and_ref.yaml")
        assert config["settings"]["sqs"]["wait_time_seconds"] == 20
        assert config["settings"]["sqs"]["region"] == "us-west-2"
        assert config["settings"]["sqs"]["queue_name"] == "dev-queue"


class TestExistingFunctionality:
    """Regression tests to ensure existing ConfigLoader behavior is preserved."""

    def test_basic_config_loading(self):
        loader = make_loader("dev")
        config = loader.load_config(FIXTURES_DIR / "base.yaml")
        assert config["sqs"]["queue_name"] == "my-queue-dev"
        assert config["sqs"]["region"] == "us-east-1"
        assert config["sqs"]["visibility_timeout"] == 300

    def test_prod_config_loading(self):
        loader = make_loader("prod")
        config = loader.load_config(FIXTURES_DIR / "base.yaml")
        assert config["sqs"]["queue_name"] == "my-queue-prod"
        assert config["outputs"]["main"]["params"]["bucket"] == "prod-bucket"

    def test_merge_dicts_unchanged(self):
        source = {"a": {"b": 1, "c": 2}}
        destination = {"a": {"b": 0, "d": 3}}
        result = merge_dicts(source, destination)
        assert result == {"a": {"b": 1, "c": 2, "d": 3}}
        assert result is destination

    def test_env_resolution_with_real_config(self):
        """Existing env:// resolution still works through the resolve decorator."""
        with patch.dict(os.environ, {"TEST_REGION": "eu-west-1"}):
            config = {"region": "env://TEST_REGION,us-east-1"}
            resolved = resolve_dict(config, logging.getLogger("test"))
            assert resolved["region"] == "eu-west-1"

    def test_env_resolution_default_value(self):
        config = {"region": "env://NONEXISTENT_VAR_12345,fallback"}
        resolved = resolve_dict(config, logging.getLogger("test"))
        assert resolved["region"] == "fallback"

    def test_caching_works(self):
        """Cache returns equivalent configs on repeated loads."""
        loader = make_loader("dev")
        config1 = loader.load_config(FIXTURES_DIR / "base.yaml")
        config2 = loader.load_config(FIXTURES_DIR / "base.yaml")
        assert config1 == config2

    def test_missing_environment_raises(self):
        loader = make_loader("nonprod")
        with pytest.raises(ConfigEnvironmentNotFoundException):
            loader.load_config(FIXTURES_DIR / "base.yaml")

    def test_missing_file_raises(self):
        loader = make_loader("dev")
        with pytest.raises(ConfigFileNotFound):
            loader.load_config("/tmp/nonexistent_config_file.yaml")
