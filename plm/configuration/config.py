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
        self.parser.add_argument("--num_epochs", type=int, default=10)
        self.parser.add_argument("--num_values", type=int, default=20)
        self.parser.add_argument("--mlm_loss", type=bool, default=False)
        self.parser.add_argument("--lossContrastiveWeight", type=float, default=0.1)
        self.parser.add_argument("--lossCorRegWeight", type=float, default=0)
        self.parser.add_argument("--simTemp", type=float, default=0.05)
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
        self.parser.add_argument("--value_categories_path", type=str,
                                 default=Path(__file__).parents[2].__str__() + "/data/value-categories.json")
        self.parser.add_argument("--train_arguments_path", type=str,
                                 default=Path(__file__).parents[2].__str__() + "/data/arguments-training.tsv")
        self.parser.add_argument("--train_labels_path", type=str,
                                 default=Path(__file__).parents[2].__str__() + "/data/labels-training.tsv")

        self.parser.add_argument("--validation_arguments_path", type=str,
                                 default=Path(__file__).parents[2].__str__() + "/data/arguments-validation.tsv")
        self.parser.add_argument("--validation_labels_path", type=str,
                                 default=Path(__file__).parents[2].__str__() + "/data/labels-validation.tsv")

        self.parser.add_argument("--zhihu_arguments_path", type=str,
                                 default=Path(__file__).parents[2].__str__() + "/data/arguments-validation-zhihu.tsv")
        self.parser.add_argument("--zhihu_labels_path", type=str,
                                 default=Path(__file__).parents[2].__str__() + "/data/labels-validation-zhihu.tsv")
        self.parser.add_argument("--test_arguments_path", type=str,
                                 default=Path(__file__).parents[2].__str__() + "/data/arguments-test.tsv")

        self.parser.add_argument("--logging_path", type=str,
                                 default=Path(__file__).parents[2].__str__() + "/assets/logs/logging.conf")
        # self.parser.add_argument("--bert_model", type=str,
        #                          default='/home/LanguageModels/mbert_base_uncased')

        self.parser.add_argument("--bert_model", type=str,
                                 default='/home/LanguageModels/deberta-large')

        self.parser.add_argument("--save_path", type=str,
                                 default=Path(__file__).parents[2].__str__() +'/deberta-large-value')


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
