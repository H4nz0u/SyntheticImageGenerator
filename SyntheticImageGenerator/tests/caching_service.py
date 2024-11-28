import unittest
from ..utilities import DataframeCacheService
import pandas as pd

import unittest

class TestDataframeCacheService(unittest.TestCase):
    def setUp(self):
        data = {
            'class': ['A', 'A', 'B', 'B'],
            'homography': ['H1', 'H2', 'H3', 'H4'],
            'background_aspect_ratio': [1.0, 1.5, 2.0, 2.5],
            'angle': [30, 45, 60, 75],
            'scaling': [0.8, 1.0, 1.2, 1.4],
            'position': [(100, 200), (150, 250), (200, 300), (250, 350)],
            'unrelated_param': ['U1', 'U2', 'U3', 'U4'],
        }
        self.df = pd.DataFrame(data)
        self.correlated_columns = [
            {'background_aspect_ratio', 'homography'},
            {'angle', 'scaling', 'position'},
        ]
        self.cache_service = DataframeCacheService(self.df, self.correlated_columns)

    def test_correlated_sampling_with_filter(self):
        filters = {'class': 'A'}
        homography = self.cache_service.sample_parameter('homography', filters)
        background_aspect_ratio = self.cache_service.sample_parameter('background_aspect_ratio', filters)
        
        # Both should come from the same row
        row_h = self.cache_service.sample_store_per_filter[self.cache_service._get_filter_key(filters)]['homography']['row']
        row_b = self.cache_service.sample_store_per_filter[self.cache_service._get_filter_key(filters)]['background_aspect_ratio']['row']
        self.assertEqual(row_h, row_b)
        self.assertEqual(self.df.at[row_h, 'homography'], homography)
        self.assertEqual(self.df.at[row_b, 'background_aspect_ratio'], background_aspect_ratio)

    def test_correlated_sampling_without_filter(self):
        homography = self.cache_service.sample_parameter('homography')
        background_aspect_ratio = self.cache_service.sample_parameter('background_aspect_ratio')
        
        # Both should come from the same row
        filter_key = self.cache_service._get_filter_key(None)
        row_h = self.cache_service.sample_store_per_filter[filter_key]['homography']['row']
        row_b = self.cache_service.sample_store_per_filter[filter_key]['background_aspect_ratio']['row']
        self.assertEqual(row_h, row_b)
        self.assertEqual(self.df.at[row_h, 'homography'], homography)
        self.assertEqual(self.df.at[row_b, 'background_aspect_ratio'], background_aspect_ratio)

    def test_multiple_correlated_sets_with_filter(self):
        filters = {'class': 'B'}
        homography = self.cache_service.sample_parameter('homography', filters)
        background_aspect_ratio = self.cache_service.sample_parameter('background_aspect_ratio', filters)
        
        angle = self.cache_service.sample_parameter('angle', filters)
        scaling = self.cache_service.sample_parameter('scaling', filters)
        position = self.cache_service.sample_parameter('position', filters)
        
        # Verify first correlated set
        row_h = self.cache_service.sample_store_per_filter[self.cache_service._get_filter_key(filters)]['homography']['row']
        row_b = self.cache_service.sample_store_per_filter[self.cache_service._get_filter_key(filters)]['background_aspect_ratio']['row']
        self.assertEqual(row_h, row_b)
        
        # Verify second correlated set
        row_a = self.cache_service.sample_store_per_filter[self.cache_service._get_filter_key(filters)]['angle']['row']
        row_s = self.cache_service.sample_store_per_filter[self.cache_service._get_filter_key(filters)]['scaling']['row']
        row_p = self.cache_service.sample_store_per_filter[self.cache_service._get_filter_key(filters)]['position']['row']
        self.assertEqual(row_a, row_s)
        self.assertEqual(row_s, row_p)
        
        # Ensure different correlated sets are from different rows (if possible)
        if len(self.cache_service.dataframe[self.cache_service.dataframe['class'] == 'B']) > 1:
            self.assertNotEqual(row_h, row_a)

    def test_independent_sampling_with_filter(self):
        filters = {'class': 'A'}
        unrelated = self.cache_service.sample_parameter('unrelated_param', filters)
        self.assertIn(unrelated, ['U1', 'U2'])
        self.assertIsNone(self.cache_service.sample_store_per_filter[self.cache_service._get_filter_key(filters)]['unrelated_param']['row'])

    def test_independent_sampling_without_filter(self):
        unrelated = self.cache_service.sample_parameter('unrelated_param')
        self.assertIn(unrelated, ['U1', 'U2', 'U3', 'U4'])
        self.assertIsNone(self.cache_service.sample_store_per_filter[self.cache_service._get_filter_key(None)]['unrelated_param']['row'])

    def test_reset_specific_filter(self):
        filters = {'class': 'A'}
        self.cache_service.sample_parameter('homography', filters)
        self.cache_service.sample_parameter('angle', filters)
        
        # Ensure samples are stored
        self.assertIn('homography', self.cache_service.sample_store_per_filter[self.cache_service._get_filter_key(filters)])
        self.assertIn('angle', self.cache_service.sample_store_per_filter[self.cache_service._get_filter_key(filters)])
        
        # Reset cache for filter 'A'
        self.cache_service.reset(filters)
        
        # Ensure cache for 'A' is cleared
        self.assertNotIn('homography', self.cache_service.sample_store_per_filter.get(self.cache_service._get_filter_key(filters), {}))
        self.assertNotIn('angle', self.cache_service.sample_store_per_filter.get(self.cache_service._get_filter_key(filters), {}))

    def test_reset_all_caches(self):
        filters_A = {'class': 'A'}
        filters_B = {'class': 'B'}
        self.cache_service.sample_parameter('homography', filters_A)
        self.cache_service.sample_parameter('angle', filters_B)
        filters_A = self.cache_service._get_filter_key(filters_A)
        filters_B = self.cache_service._get_filter_key(filters_B)
        # Ensure samples are stored
        self.assertIn('homography', self.cache_service.sample_store_per_filter[filters_A])
        self.assertIn('angle', self.cache_service.sample_store_per_filter[filters_B])
        
        # Reset all caches
        self.cache_service.reset()
        
        # Ensure all caches are cleared
        self.assertEqual(len(self.cache_service.sample_store_per_filter), 0)
        self.assertEqual(len(self.cache_service.group_row_map_per_filter), 0)

    def test_conflicting_correlations(self):
        # Attempt to assign a column to multiple correlated sets
        conflicting_correlated_columns = [
            {'background_aspect_ratio', 'homography'},
            {'homography', 'angle'},  # 'homography' is already in the first set
        ]
        with self.assertRaises(ValueError):
            DataframeCacheService(self.df, conflicting_correlated_columns)

    def test_no_matching_filters(self):
        with self.assertRaises(ValueError):
            self.cache_service.sample_parameter('homography', {'class': 'C'})  # 'C' does not exist

    def test_filter_key_creation(self):
        filters = {'class': 'A', 'another_key': 'value'}
        filter_key = self.cache_service._get_filter_key(filters)
        expected_key = frozenset([('class', 'A'), ('another_key', 'value')])
        self.assertEqual(filter_key, expected_key)

if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)
