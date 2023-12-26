import unittest
from utils import chunk_file_list 

class TestChunkFileList(unittest.TestCase):

    def test_normal_continuity(self):
        files = ["file_20230101T120000", "file_20230101T130000", "file_20230101T140000"]
        chunks = chunk_file_list(files, 60, 5)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks, [files])

    def test_greater_discontinuity(self):
        files = ["file_20230101T120000", "file_20230101T130000", "file_20230101T141000"]
        chunks = chunk_file_list(files, 60, 5)
        self.assertEqual(len(chunks), 2)

    def test_lesser_discontinuity(self):
        files = ["file_20230101T120000", "file_20230101T125000", "file_20230101T140000"]
        chunks = chunk_file_list(files, 60, 5)
        self.assertEqual(len(chunks), 3)

    def test_multiple_discontinuities(self):
        files = ["file_20230101T120000", "file_20230101T125000", "file_20230101T141000", "file_20230101T151000"]
        chunks = chunk_file_list(files, 60, 5)
        print(chunks)
        self.assertEqual(len(chunks), 3)

    def test_empty_list(self):
        files = []
        chunks = chunk_file_list(files, 60, 5)
        self.assertEqual(len(chunks), 0)

    def test_single_file(self):
        files = ["file_20230101T120000"]
        chunks = chunk_file_list(files, 60, 5)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks, [files])

if __name__ == '__main__':
    unittest.main()