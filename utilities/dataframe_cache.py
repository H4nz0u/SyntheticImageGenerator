from .logging import logger
dataframe_cache = {}
import pandas as pd

def get_cached_dataframe(name, dataframe_path=None):
    if name not in dataframe_cache:
        if dataframe_path is None:
            raise ValueError(f"DataFrame '{name}' not found in cache and no path provided")
        try:
            dataframe_cache[name] = pd.read_pickle(dataframe_path)
            logger.info(f"Loaded and cached DataFrame '{name}' from {dataframe_path}")
        except Exception as e:
            logger.error(f"Failed to load DataFrame from {dataframe_path}: {e}")
            raise
    return dataframe_cache[name]