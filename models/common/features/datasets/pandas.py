"""Module with Pandas dataset implementation."""

from typing import Any

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class PandasDataset(Dataset):
    """Pandas dataset implementation."""
    def __init__(
        self,
        dataframe: pd.DataFrame,
        return_dicts: bool = False,
        target_col: str | None = None,
        mask_value: int | None = None,
        masking_proba: float | None = None,
        columns_to_mask: list[str] | None = None,
    ):
        """Instantiate dataset.

        Args:
            dataframe: input dataframe to wrap as toch dataset
            return_dicts: flag to return dict or tensor of values
            target_col: target column name
            mask_value: value to mask category columns
            masking_proba: probabilty to mask category values mask value
            columns_to_mask: columns to apply masking
        """
        self.dataframe = dataframe
        self.return_dicts = return_dicts
        self.target_col = target_col
        self.mask_value = mask_value
        self.masking_proba = masking_proba
        self.columns_to_mask = columns_to_mask or []

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

        # apply masking
        if self.mask_value is not None and self.masking_proba is not None:
            for col in self.columns_to_mask:
                if np.random.uniform() < self.masking_proba:
                    # to avoid SettingWithCopyWarning
                    row = row.where(row.index != col, self.mask_value)

        if self.return_dicts:
            return {k: v if not isinstance(v, list | tuple) else np.array(v) for k, v in row.to_dict().items()}

        if not self.target_col:
            return row.values

        return row.drop(self.target_col).values, row[self.target_col]
