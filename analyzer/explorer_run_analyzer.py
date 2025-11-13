import json
import os

import pandas as pd


class ExplorerRunAnalyzer:
    def __init__(self, log_dir: str, csv_name: str = "explorer_summary.csv"):
        self.log_dir = log_dir
        self.csv_path = os.path.join(self.log_dir, csv_name)
        self.columns = [
            "timestamp",
            "model_name",
            "env_name",
            "instruction",
            "action_path",
            "step_count",
            "final_score",
        ]
        self.df = pd.DataFrame(columns=self.columns)

    def record_run(
        self,
        timestamp: str,
        model_name: str,
        env_name: str,
        instruction: str,
        action_path: list,
        step_count: int,
        final_score,
    ):
        self.df.loc[len(self.df)] = [
            timestamp,
            model_name,
            env_name,
            instruction,
            action_path,
            step_count,
            final_score,
        ]

    def save_to_csv(self):
        if self.df.empty:
            return
        temp_df = self.df.copy()
        temp_df["action_path"] = temp_df["action_path"].apply(
            lambda path: json.dumps(path, ensure_ascii=False)
        )
        header = not os.path.exists(self.csv_path)
        temp_df.to_csv(
            self.csv_path,
            mode="a",
            header=header,
            index=False,
            encoding="utf-8",
        )
        self.df = self.df.iloc[0:0]

