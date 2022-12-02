"""
    script for setting config for all the parameters and paths
"""
# ========================================================
import argparse
from pathlib import Path


# ========================================================
class BaseConfig:
    """
        BaseConfig:
    """

    def __init__(self):
        """
            Init Configs
        """
        self.parser = argparse.ArgumentParser()

    def _variables(self):
        self.parser.add_argument("--device", type=str, default="cuda:1")
        self.parser.add_argument("--batch_size", type=int, default=8)
        self.parser.add_argument("--shuffle", type=int, default=True)
        self.parser.add_argument("--num_epochs", type=int, default=20)
        self.parser.add_argument("--num_intents", type=int, default=60)
        self.parser.add_argument("--level", type=str, default="2")

    def _paths(self) -> None:
        """
            function to add path
        Returns:
            None
        """
        self.parser.add_argument("--data_path", type=str,
                                 default=Path(__file__).parents[2].__str__() + "/data")
        self.parser.add_argument("--values_path", type=str,
                                 default=Path(__file__).parents[2].__str__() + "/data/values.json")
        self.parser.add_argument("--arguments_path", type=str,
                                 default=Path(__file__).parents[2].__str__() + "/data/arguments-training.tsv")
        self.parser.add_argument("--labels_path", type=str,
                                 default=Path(__file__).parents[2].__str__() + "/data/labels.training.tsv")
        self.parser.add_argument("--logging_path", type=str,
                                 default=Path(__file__).parents[2].__str__() + "/assets/logs/logging.conf")
        self.parser.add_argument("--bert_model", type=str,
                                 default='/home/LanguageModels/mbert_base_uncased')
        self.parser.add_argument("--saved_models_path", type=str,
                                 default=Path(__file__).parents[2].__str__() + "/assets/saved")

    def get_config(self) -> argparse:
        """
            Get Base Configurations
        Returns:
            configs: Return Configs
        """
        self._paths()
        self._variables()
        configs = self.parser.parse_args()
        return configs
