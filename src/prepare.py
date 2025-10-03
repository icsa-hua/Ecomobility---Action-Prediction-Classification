import pandas as pd
from sklearn.model_selection import train_test_split

class DatasetPrep:
    def __init__(self, csv_file, seed=42):
        self.csv_file = csv_file
        self.seed = seed

    def one_hot_to_class(self, row):
        if row['fwd'] == 1:
            return 0
        elif row['left'] == 1:
            return 1
        elif row['right'] == 1:
            return 2
        elif row['stop'] == 1:
            return 3
        else:
            raise ValueError(f"Invalid one-hot row: {row}")

    def prepare(self):
        df = pd.read_csv(self.csv_file)
        df['label'] = df.apply(self.one_hot_to_class, axis=1)
        df = df[['file', 'label']]

        train_val_df, test_df = train_test_split(
            df, test_size=0.10, random_state=self.seed, stratify=df['label']
        )
        train_df, val_df = train_test_split(
            train_val_df, test_size=0.15/0.90, random_state=self.seed, stratify=train_val_df['label']
        )
        return {"train": train_df, "val": val_df, "test": test_df}
