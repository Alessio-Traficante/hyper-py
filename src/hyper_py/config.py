
import yaml
import os

class HyperConfig:
    def __init__(self, yaml_file):
        if not os.path.exists(yaml_file):
            raise FileNotFoundError(f"Configuration file '{yaml_file}' not found.")
        with open(yaml_file, 'r') as f:
            self.config = yaml.safe_load(f)
        self.validate()

    def get(self, section, key=None, default=None):
        if key:
            return self.config.get(section, {}).get(key, default)
        return self.config.get(section, default)

    def validate(self):
        required_sections = [
            'paths', 'units', 'control',
            'survey', 'detection', 'photometry',
            'background'
        ]
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required config section: '{section}'")

        # Example check
        if self.get('control', 'use_this_rms') and self.get('control', 'this_rms_value') is None:
            raise ValueError("If 'use_this_rms' is true, 'this_rms_value' must be defined.")
            
            
    def to_dict(self):
        """Return the raw config dictionary for serialization or multiprocessing."""
        return self.config

    @staticmethod
    def from_dict(config_dict):
        """Create a HyperConfig object from an existing dictionary."""
        cfg = HyperConfig.__new__(HyperConfig)
        cfg.config = config_dict
        cfg.validate()
        return cfg         
