from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod
from pathlib import Path
from ..utils.config_loader import load_yaml


class BaseProvider(ABC):
    """Base class for model providers."""

    def __init__(self, model_name: str, config: Dict[str, Any]):
        self.model_name = model_name
        self.config = config
        self.model_config = config.get("models", {}).get(model_name, {})
        self.batch_config = config.get("batch", {})

    @abstractmethod
    def infer(self, prompt: str, **kwargs) -> str:
        pass

    def batch_infer(self, prompts: List[str], **kwargs) -> List[str]:
        """Default batch inference implementation."""
        results = []
        batch_size = self.batch_config.get("size", 20)

        for i in range(0, len(prompts), batch_size):
            batch = prompts[i : i + batch_size]
            for prompt in batch:
                results.append(self.infer(prompt, **kwargs))
        return results


class AzureOpenAIProvider(BaseProvider):
    """Azure OpenAI provider."""

    def __init__(self, model_name: str, config: Dict[str, Any]):
        super().__init__(model_name, config)
        try:
            from openai import AzureOpenAI

            self.client = AzureOpenAI(
                azure_endpoint=config.get("endpoint", ""),
                api_key=config.get("api_key", ""),
                api_version=config.get("api_version", "2024-10-21"),
            )
        except ImportError:
            raise ImportError(
                "Azure OpenAI is not available. Please install openai package or use OpenAI provider instead."
            )

    def infer(self, prompt: str, **kwargs) -> str:
        deployment = self.model_config.get("deployment", self.model_name)

        # Build params without spreading model_config directly
        params = {
            "model": deployment,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.model_config.get("temperature", 0.0),
            "max_tokens": self.model_config.get("max_tokens", 4096),
            **kwargs,
        }

        response = self.client.chat.completions.create(**params)
        return response.choices[0].message.content


class OpenAIProvider(BaseProvider):
    """OpenAI provider."""

    def __init__(self, model_name: str, config: Dict[str, Any]):
        super().__init__(model_name, config)
        try:
            from openai import OpenAI
            import os

            # Get API key from config or environment
            api_key = config.get("api_key")
            if (
                isinstance(api_key, str)
                and api_key.startswith("${")
                and api_key.endswith("}")
            ):
                env_var = api_key[2:-1]
                api_key = os.getenv(env_var)

            # Fallback to environment variable if not in config
            if not api_key:
                api_key = os.getenv("OPENAI_API_KEY")

            self.client = OpenAI(api_key=api_key)
        except ImportError:
            raise ImportError(
                "OpenAI package not installed. Please run: pip install openai"
            )

    def infer(self, prompt: str, **kwargs) -> str:
        # Build params carefully - only include valid OpenAI parameters
        params = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.model_config.get("temperature", 0.0),
            "max_tokens": self.model_config.get("max_tokens", 4096),
        }

        # Add optional parameters if they exist
        if "top_p" in self.model_config:
            params["top_p"] = self.model_config["top_p"]
        if "frequency_penalty" in self.model_config:
            params["frequency_penalty"] = self.model_config["frequency_penalty"]
        if "presence_penalty" in self.model_config:
            params["presence_penalty"] = self.model_config["presence_penalty"]

        # Handle special parameters for advanced models
        if "reasoning_effort" in self.model_config:
            params["reasoning_effort"] = self.model_config["reasoning_effort"]
        if "reasoning_strategy" in self.model_config:
            params["reasoning_strategy"] = self.model_config["reasoning_strategy"]
        if "reasoning_depth" in self.model_config:
            params["reasoning_depth"] = self.model_config["reasoning_depth"]

        # Add any additional kwargs that are valid
        for key, value in kwargs.items():
            if key not in params and key not in [
                "deployment"
            ]:  # Filter out Azure-specific params
                params[key] = value

        response = self.client.chat.completions.create(**params)
        return response.choices[0].message.content


class SimpleModelProvider:
    """Simplified provider that uses OpenAI by default"""

    def __init__(self, config: Optional[Dict] = None):
        import os
        from openai import OpenAI

        self.config = config or {}
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Please set OPENAI_API_KEY environment variable")

        self.client = OpenAI(api_key=api_key)
        self.model_name = self.config.get("model_name", "gpt-4o-mini")

    def infer(self, prompt: str, model_name: Optional[str] = None, **kwargs) -> str:
        """Simple inference using OpenAI"""
        model = model_name or self.model_name

        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.config.get("temperature", 0.0),
            max_tokens=self.config.get("max_tokens", 4096),
        )
        return response.choices[0].message.content

    def batch_infer(
        self, prompts: List[str], model_name: Optional[str] = None
    ) -> List[str]:
        """Batch inference"""
        return [self.infer(p, model_name) for p in prompts]


class ModelRouter:
    """Routes model requests to appropriate providers."""

    def __init__(self, config_dir: str = "config/model_params"):
        self.config_dir = Path(config_dir)
        self.providers = {}

        # Check if config directory exists
        if self.config_dir.exists():
            config_file = self.config_dir / "model_configs.yaml"
            if config_file.exists():
                self.main_config = load_yaml(config_file)
                self._load_providers()
            else:
                # Default configuration
                self.main_config = {
                    "routing": {
                        "default_provider": "openai",
                        "patterns": {"gpt": "openai", "o1": "openai", "o3": "openai"},
                    },
                    "providers": {
                        "openai": {
                            "class": "OpenAIProvider",
                            "config_file": "openai.yaml",
                        }
                    },
                }
        else:
            # Minimal default config
            self.main_config = {
                "routing": {"default_provider": "openai", "patterns": {}},
                "providers": {},
            }

    def _load_providers(self):
        """Load all provider configurations."""
        for provider_name, provider_info in self.main_config.get(
            "providers", {}
        ).items():
            config_file = self.config_dir / provider_info["config_file"]
            if config_file.exists():
                self.providers[provider_name] = {
                    "class": provider_info["class"],
                    "config": load_yaml(config_file),
                }

    def get_provider(self, model_name: str) -> BaseProvider:
        """Get appropriate provider for model."""
        # Check patterns first
        patterns = self.main_config.get("routing", {}).get("patterns", {})
        provider_name = None

        for pattern, provider in patterns.items():
            if pattern in model_name:
                provider_name = provider
                break

        # Use default if no pattern matched
        if not provider_name:
            provider_name = self.main_config.get("routing", {}).get(
                "default_provider", "openai"
            )

        # Get provider class and config
        if provider_name in self.providers:
            provider_info = self.providers[provider_name]
            provider_class = self._get_provider_class(provider_info["class"])
            return provider_class(model_name, provider_info["config"])
        else:
            # Fallback to simple OpenAI config
            return OpenAIProvider(
                model_name,
                {
                    "api_key": "${OPENAI_API_KEY}",
                    "models": {model_name: {"temperature": 0.0, "max_tokens": 4096}},
                },
            )

    def _get_provider_class(self, class_name: str):
        """Get provider class by name."""
        classes = {
            "AzureOpenAIProvider": AzureOpenAIProvider,
            "OpenAIProvider": OpenAIProvider,
        }
        return classes.get(class_name, OpenAIProvider)


class ModelProvider:
    """Main interface for model inference - with config support."""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}

        # Check if we should use simple mode based on settings
        provider_type = self.config.get("provider", "").lower()

        # For simple OpenAI setup, use SimpleModelProvider
        if provider_type == "openai" and not Path("config/model_params").exists():
            self.simple_mode = True
            self.provider = SimpleModelProvider(config)
        # If model_params directory exists, use the router
        elif Path("config/model_params").exists():
            self.simple_mode = False
            self.router = ModelRouter()
            # Override default provider if specified in settings
            if provider_type:
                self.router.main_config["routing"]["default_provider"] = provider_type
        else:
            # Fallback to simple mode
            self.simple_mode = True
            self.provider = SimpleModelProvider(config)

    def infer(self, prompt: str, model_name: Optional[str] = None, **kwargs) -> str:
        """Perform inference with automatic provider selection."""
        if self.simple_mode:
            return self.provider.infer(prompt, model_name, **kwargs)
        else:
            model = model_name or self.config.get("model_name", "gpt-4o-mini")
            provider = self.router.get_provider(model)
            return provider.infer(prompt, **kwargs)

    def batch_infer(
        self, prompts: List[str], model_name: Optional[str] = None
    ) -> List[str]:
        """Batch inference"""
        if self.simple_mode:
            return self.provider.batch_infer(prompts, model_name)
        else:
            model = model_name or self.config.get("model_name", "gpt-4o-mini")
            provider = self.router.get_provider(model)
            return provider.batch_infer(prompts)
