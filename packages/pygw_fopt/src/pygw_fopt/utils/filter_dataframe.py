import pandas as pd
import numpy as np

def filter_dataframe(df, filter_dict):
	mask = pd.Series(True, index=df.index)
	for col, value in filter_dict.items():
		if col not in df.columns:
			raise ValueError(f"Column {col} not found in DataFrame")
		mask &= np.isclose(df[col], value, rtol=1e-5, atol=1e-8)
	
	filtered_df = df[mask]
	return filtered_df
