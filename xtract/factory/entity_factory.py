from pathlib import Path
from typing import List, Dict, Any
from ..utils.config_loader import load_yaml

class Entity:
    def __init__(self, name: str, config: Dict[str, Any], evaluations: Dict[str, Any]):
        self.name = name
        self.prompt = config.get("prompt", "")
        self.context = config.get("context", {})
        self.format = config.get("format", {})
        self.model_override = config.get("model", None)
        self.enabled = config.get("enabled", True)
        self.evaluations = config.get("evaluations", {})
        self.available_evaluations = evaluations
        
    def format_prompt(self, content: str) -> str:
        context_vars = {"content": content}
        context_vars.update(self.context)
        return self.prompt.format(**context_vars)
    
    def get_active_evaluations(self) -> Dict[str, Dict]:
        """Return only enabled evaluations for this entity"""
        active = {}
        for eval_name, enabled in self.evaluations.items():
            if enabled and eval_name in self.available_evaluations:
                active[eval_name] = self.available_evaluations[eval_name]
        return active

class EntityFactory:
    def __init__(self, extraction_dir: str = "config/extraction", evaluation_dir: str = "config/evaluation"):
        self.entities = []
        self.evaluations = self._load_evaluations(evaluation_dir)
        self._load_entities(extraction_dir)
    
    def _load_evaluations(self, directory: str) -> Dict[str, Any]:
        """Load all evaluation configurations"""
        evaluations = {}
        path = Path(directory)
        if path.exists():
            for yaml_file in path.glob("*.yaml"):
                config = load_yaml(yaml_file)
                if config.get("enabled", True):
                    evaluations[yaml_file.stem] = config
        return evaluations
    
    def _load_entities(self, directory: str):
        """Load entities from YAML files"""
        path = Path(directory)
        if not path.exists():
            return
            
        for yaml_file in path.glob("*.yaml"):
            config = load_yaml(yaml_file)
            if config.get("enabled", True):
                entity = Entity(yaml_file.stem, config, self.evaluations)
                self.entities.append(entity)
    
    def get_entities(self) -> List[Entity]:
        return [e for e in self.entities if e.enabled]
    
    def get_entity_evaluations(self, entity_name: str) -> Dict[str, Any]:
        """Get evaluations for specific entity"""
        for entity in self.entities:
            if entity.name == entity_name:
                return entity.evaluations
        return {}