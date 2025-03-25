from dataclasses import dataclass
import os


@dataclass
class Configuration:
    openai_api_key: str
    serpapi_api_key: str
    max_task_depth: int
    model_settings: dict[str, "ModelConfiguration"]

    def load_model_settings_from_yaml(self, yaml_path: str) -> None:
        with open(yaml_path, "r") as f:
            yaml_data = yaml.load(f)
        self.model_settings = yaml_data["model_settings"]


@dataclass
class ModelConfiguration:
    model: str
    system_instructions: str


config: Configuration | None = None


def get_configuration() -> Configuration:
    global config
    if config is not None:
        return config
    config = Configuration(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        serpapi_api_key=os.getenv("SERPAPI_API_KEY"),
        max_task_depth=os.getenv("MAX_TASK_DEPTH"),
    )
    return config
