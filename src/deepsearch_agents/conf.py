from dataclasses import dataclass, field
import os
from typing import Literal
from agents import ModelSettings
import yaml

from deepsearch_agents.log import logger


@dataclass
class ModelConfig:
    """
    Configuration class for LLM model settings.

    This class holds the configuration parameters for different language models,
    including temperature, token limits, and other model-specific settings.
    """

    model_name: str
    """Name of the model (e.g., 'gpt-4', 'gpt-3.5-turbo')"""

    max_tokens: int
    """Maximum number of tokens to generate in the response"""

    temperature: float | None = None
    """Controls randomness in the model's output (0.0 to 2.0)"""

    top_p: float | None = None
    """Controls diversity via nucleus sampling (0.0 to 1.0)"""

    tool_choice: str | None = None
    """The tool choice to use when calling the model."""

    parallel_tool_calls: bool | None = None
    """Whether to use parallel tool calls when calling the model."""

    def as_model_settings(self) -> ModelSettings:
        """
        Converts ModelConfig to ModelSettings format used by the agents framework.

        Returns:
            ModelSettings: Configuration in the agents framework format
        """
        return ModelSettings(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            tool_choice=self.tool_choice,
            parallel_tool_calls=self.parallel_tool_calls,
        )


@dataclass
class ExecutionConfig:
    """
    Configuration class for execution settings.
    """

    max_task_depth: int = 2
    """Maximum depth for nested task execution"""

    max_token_usage: int = 100_000
    """Maximum token usage for the execution"""

    max_turns: int = 15
    """Maximum number of turns for the execution"""

    max_critical_attempts: int = 2
    """Maximum number of critical attempts for the execution"""


@dataclass
class Configuration:
    """
    Main configuration class for the DeepSearch Agents application.

    This class manages all configuration settings including API keys,
    model configurations, and application parameters.
    """

    openai_base_url: str
    """Base URL for OpenAI API endpoints"""

    openai_api_key: str
    """Primary OpenAI API key"""

    tracing_openai_api_key: str
    """OpenAI API key for tracing"""

    jina_api_key: str
    """Jina API key"""

    serpapi_api_key: str
    """API key for SerpAPI service"""

    execution_config: ExecutionConfig = field(default_factory=ExecutionConfig)
    """Configuration for execution settings"""

    model_settings: dict[str, ModelConfig] | None = None
    """Dictionary of model configurations indexed by model name"""

    def load_model_settings_from_yaml(self, yaml_path: str):
        """
        Loads model configurations from a YAML file.

        Args:
            yaml_path: Path to the YAML configuration file
        """
        if self.model_settings:
            logger.warning("Model settings already loaded, skipping")
            return
        with open(yaml_path, "r") as f:
            yaml_data = yaml.safe_load(f)
        self.model_settings = {
            model_name: ModelConfig(**model_config)
            for model_name, model_config in yaml_data["models"].items()
        }
        self.execution_config = ExecutionConfig(**yaml_data["execution"])

    def get_model_config(self, model_name: str) -> ModelConfig:
        """
        Retrieves configuration for a specific model.

        Args:
            model_name: Name of the model to get configuration for

        Returns:
            ModelConfig: Configuration for the specified model

        Raises:
            ValueError: If model settings are not loaded or model not found
        """
        if not self.model_settings:
            raise ValueError("Model settings not loaded")
        return self.model_settings[model_name]


# Global configuration instance
config: Configuration | None = None


def get_configuration() -> Configuration:
    """
    Gets or creates the global configuration instance.

    This function implements the singleton pattern for configuration management.
    It loads environment variables and model settings from YAML on first call.

    Returns:
        Configuration: The global configuration instance
    """
    global config
    if config is not None:
        return config

    from dotenv import load_dotenv

    load_dotenv()
    config = Configuration(
        openai_base_url=os.getenv("OPENAI_BASE_URL", ""),
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        tracing_openai_api_key=os.getenv("TRACING_OPENAI_API_KEY", ""),
        jina_api_key=os.getenv("JINA_API_KEY", ""),
        serpapi_api_key=os.getenv("SERPAPI_API_KEY", ""),
    )
    config.load_model_settings_from_yaml(os.path.join("settings.yaml"))
    return config


if __name__ == "__main__":
    print(get_configuration())
    print(get_configuration().get_model_config("summary"))
