import unittest
from utils.io import list_files, get_image_label_pairs


class TestIO(unittest.TestCase):
    def test_list_files(self):

        self.assertIsNotNone(list_files('data/middle-ear-dataset/aom', ''))
        self.assertEqual(len(list_files('data/middle-ear-dataset/aom', '')), 119)
        
       # self.assertIsNotNone(list_files('data\middle-ear-dataset/am', ''))
        
    def test_get_image_label_pairs(self):
        self.assertIsNotNone(get_image_label_pairs('data/middle-ear-dataset/aom', 'aom'))
        self.assertEqual(len(get_image_label_pairs('data/middle-ear-dataset/aom', 'aom')[0]), 119)
        
    