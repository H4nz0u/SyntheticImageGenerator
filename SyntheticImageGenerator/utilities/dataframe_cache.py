from .logging import logger
import pandas as pd
import ast
import random
from typing import List, Set, Optional, Dict, Any, FrozenSet

class DataframeCacheService:
    def __init__(self, dataframe: pd.DataFrame, correlated_columns: Optional[List[Set[str]]] = None):
        """
        Initializes the caching service.

        :param dataframe: pandas DataFrame containing transformation parameters.
        :param correlated_columns: List of sets, each containing column names that are correlated and should be sampled from the same row.
        """
        self.dataframe = dataframe.reset_index(drop=True)
        self.correlated_columns = correlated_columns or []
        self.column_to_group = self._map_columns_to_correlated_sets(self.correlated_columns)
        
        # Initialize separate caches for each unique filter
        self.sample_store_per_filter: Dict[FrozenSet[tuple], Dict[str, Dict[str, Any]]] = {}
        self.group_row_map_per_filter: Dict[FrozenSet[tuple], Dict[FrozenSet[str], int]] = {}

    def _map_columns_to_correlated_sets(self, correlated_columns: List[Set[str]]) -> Dict[str, FrozenSet[str]]:
        """
        Creates a mapping from column names to their correlated column sets.

        :param correlated_columns: List of sets containing correlated column names.
        :return: Dictionary mapping each column name to its correlated set.
        """
        mapping = {}
        for group in correlated_columns:
            frozen_group = frozenset(group)
            for column in group:
                if column in mapping:
                    raise ValueError(f"Column '{column}' is assigned to multiple correlated sets.")
                mapping[column] = frozen_group
        return mapping

    def _get_filter_key(self, filters: Optional[Dict[str, Any]]) -> FrozenSet[tuple]:
        """
        Converts the filters dictionary into a frozenset of tuples to use as a dictionary key.

        :param filters: Dictionary of filter criteria.
        :return: Frozenset of filter key-value tuples.
        """
        if filters is None:
            return frozenset()
        return frozenset(filters.items())

    def sample_parameter(self, key: str, filters: Optional[Dict[str, Any]] = None):
        """
        Samples a parameter value, considering correlations and filter criteria.

        :param key: The parameter key to sample.
        :param filters: Optional dictionary of filter criteria (e.g., {'class': 'A'}).
        :return: Sampled value.
        """
        filter_key = self._get_filter_key(filters)

        # Initialize caches for this filter if not already present
        if filter_key not in self.sample_store_per_filter:
            self.sample_store_per_filter[filter_key] = {}
        if filter_key not in self.group_row_map_per_filter:
            self.group_row_map_per_filter[filter_key] = {}

        sample_store = self.sample_store_per_filter[filter_key]
        group_row_map = self.group_row_map_per_filter[filter_key]

        # If the parameter is already sampled, return its value
        if key in sample_store:
            return sample_store[key]['value']

        # Check if the parameter is part of any correlated set
        correlated_set = self.column_to_group.get(key)
        if correlated_set:
            if correlated_set in group_row_map:
                row_idx = group_row_map[correlated_set]
            else:
                filtered_df = self._apply_filters(filters)
                row_idx = self._select_random_row(filtered_df)
                group_row_map[correlated_set] = row_idx

            # Sample all parameters in the correlated set from the same row
            for column in correlated_set:
                if column not in sample_store:
                    sampled_value = self.dataframe.at[row_idx, column]
                    sample_store[column] = {'row': row_idx, 'value': sampled_value}

            return sample_store[key]['value']
        else:
            # Parameter is not part of any correlated set; sample independently within the filtered DataFrame
            sampled_value = self._random_sample(key, filters)
            sample_store[key] = {'row': None, 'value': sampled_value}
            return sampled_value

    def _apply_filters(self, filters: Optional[Dict[str, Any]]) -> pd.DataFrame:
        """
        Applies filter criteria to the DataFrame and returns the filtered DataFrame.

        :param filters: Dictionary of filter criteria.
        :return: Filtered pandas DataFrame.
        """
        if not filters:
            return self.dataframe
        filtered_df = self.dataframe
        for f_key, f_val in filters.items():
            if f_key not in filtered_df.columns:
                raise ValueError(f"Filter key '{f_key}' does not exist in the DataFrame.")
            filtered_df = filtered_df[filtered_df[f_key] == f_val]
        if filtered_df.empty:
            raise ValueError(f"No rows match the provided filters: {filters}")
        return filtered_df

    def _select_random_row(self, filtered_df: pd.DataFrame) -> int:
        """
        Selects a random row index from the filtered DataFrame.

        :param filtered_df: The filtered pandas DataFrame to sample from.
        :return: Random row index from the original DataFrame.
        """
        return filtered_df.sample(n=1).index[0]

    def _random_sample(self, key: str, filters: Optional[Dict[str, Any]] = None):
        """
        Samples a random value for the given key from the filtered DataFrame.

        :param key: The parameter key to sample.
        :param filters: Optional dictionary of filter criteria.
        :return: Sampled value.
        """
        filtered_df = self._apply_filters(filters)
        sampled_value = filtered_df[key].sample(n=1).iloc[0]
        return sampled_value

    def reset(self, filters: Optional[Dict[str, Any]] = None):
        """
        Resets the sample store and group-row mappings for a new run, optionally for specific filters.

        :param filters: Optional dictionary of filter criteria to reset specific caches.
        """
        if filters:
            filter_key = self._get_filter_key(filters)
            self.sample_store_per_filter.pop(filter_key, None)
            self.group_row_map_per_filter.pop(filter_key, None)
        else:
            # Reset all caches
            self.sample_store_per_filter = {}
            self.group_row_map_per_filter = {}
            logger.info("All caches have been reset.")

dataframe_cache = {}

def get_cached_dataframe(name: str, dataframe_path: Optional[str] = None, correlated_columns: Optional[List[List[str]]] = None) -> DataframeCacheService:
    """
    Retrieves a cached DataframeCacheService instance by name. If not cached, loads it from the provided path.

    :param name: The name identifier for the cached DataFrame.
    :param dataframe_path: The file path to load the DataFrame from if not cached.
    :param correlated_columns: Optional list of correlated column sets to initialize the service.
    :return: An instance of DataframeCacheService.
    """
    if name not in dataframe_cache:
        if dataframe_path is None:
            raise ValueError(f"DataFrame '{name}' not found in cache and no path provided")
        try:
            dataframe = pd.read_csv(dataframe_path)

            # Parse the 'transformation_matrix' column as a Python list if it exists
            if 'transformation_matrix' in dataframe.columns:
                dataframe['transformation_matrix'] = dataframe['transformation_matrix'].apply(ast.literal_eval)
            
            # Convert lists to sets for correlated_columns
            correlated_columns_sets = [set(group) for group in correlated_columns] if correlated_columns else []
            
            # Initialize the DataframeCacheService with correlated_columns
            dataframe_cache[name] = DataframeCacheService(dataframe, correlated_columns_sets)
            
            logger.info(f"Loaded and cached DataFrame '{name}' from '{dataframe_path}'")
        except Exception as e:
            logger.error(f"Failed to load DataFrame '{name}' from '{dataframe_path}': {e}")
            raise
    return dataframe_cache[name]
