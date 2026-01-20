from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Union

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Between:
    low: float
    high: float


@dataclass(frozen=True)
class In:
    values: Iterable[Any]


@dataclass(frozen=True)
class GreaterThan:
    threshold: float


@dataclass(frozen=True)
class SmallerThan:
    threshold: float


@dataclass(frozen=True)
class Contains:
    """Unordered membership. For strings, treats as substring.
    For lists, checks if all 'values' exist anywhere in the target."""

    values: Any


@dataclass(frozen=True)
class ContainsOrdered:
    """Ordered membership. For strings, treats as substring.
    For lists, checks if the exact sub-sequence exists in order."""

    sequence: Union[str, Iterable[Any]]


class Constraint:
    Between = Between
    In = In
    GreaterThan = GreaterThan
    SmallerThan = SmallerThan
    Contains = Contains
    ContainsOrdered = ContainsOrdered

    Type = Union[
        Between,
        In,
        GreaterThan,
        SmallerThan,
        Contains,
        ContainsOrdered,
        float,
        int,
        str,
        Any,
    ]


def filter_dataframe(
    df: pd.DataFrame, filter_dict: Dict[str, Constraint.Type]
) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)

    for col, crit in filter_dict.items():
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found.")

        col_series = df[col]

        match crit:
            case Constraint.Between(l, h):
                mask &= (col_series >= l) & (col_series <= h)

            case Constraint.In(v):
                mask &= col_series.isin(v)

            case Constraint.GreaterThan(t):
                mask &= col_series > t

            case Constraint.SmallerThan(t):
                mask &= col_series < t

            case Constraint.Contains(v):
                if isinstance(v, (str, int, float)):  # Single element check
                    mask &= col_series.apply(
                        lambda x: v in x if isinstance(x, Iterable) else False
                    )
                else:  # Set-based check (Subset)
                    v_set = set(v)
                    mask &= col_series.apply(
                        lambda x: v_set.issubset(set(x))
                        if isinstance(x, Iterable)
                        else False
                    )

            case Constraint.ContainsOrdered(seq):
                if isinstance(seq, str):
                    mask &= col_series.str.contains(seq, na=False, regex=False)
                else:
                    # Sub-sequence matching for lists
                    def is_subsequence(main_list, sub):
                        n, m = len(main_list), len(sub)
                        return any(
                            all(main_list[i + j] == sub[j] for j in range(m))
                            for i in range(n - m + 1)
                        )

                    mask &= col_series.apply(
                        lambda x: is_subsequence(list(x), list(seq))
                        if isinstance(x, Iterable)
                        else False
                    )

            case _ if pd.api.types.is_numeric_dtype(col_series) and isinstance(
                crit, (int, float, np.number)
            ):
                mask &= np.isclose(col_series, crit, rtol=1e-5, atol=1e-8)

            case _:
                mask &= col_series == crit

    return df[mask]
