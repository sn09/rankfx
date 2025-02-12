"""Module with Pandas dataset implementation."""

from typing import Any

import pandas as pd
from torch.utils.data import Dataset


class PandasDataset(Dataset):
    """Pandas dataset implementation."""
    def __init__(
        self,
        dataframe: pd.DataFrame,
        return_dicts: bool = False,
        target_col: str | None = None,
    ):
        """Instantiate dataset.

        Args:
            dataframe: input dataframe to wrap as toch dataset
            return_dicts: flag to return dict or tensor of values
            target_col: target column name
        """
        self.dataframe = dataframe
        self.return_dicts = return_dicts
        self.target_col = target_col

    def __len__(self) -> int:
        """Get dataset length.

        Returns:
            Dataset length
        """
        return len(self.dataframe)

    def  __getitem__(self, index: int) -> dict[str, Any] | Any:
        """Get dataset item by index.

        Args:
            index: index to take

        Returns:
            Dataset element under specified index
        """
        row = self.dataframe.iloc[index]
        if self.return_dicts:
            return row.to_dict()

        if not self.target_col:
            return row.values

        return row.drop(self.target_col).values, row[self.target_col]
