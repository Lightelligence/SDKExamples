import logging
import os
import shutil
import tempfile
import unittest

from benchmarks import efficientnet_b0, mnist_cnn, mobilenet_v2, resnet_50

TMP_DIR = os.getenv("TMPDIR") or ("/fast/algo/tmp"
                                  if os.path.exists("/fast/algo/tmp") else "/tmp")

BENCHMARK_DATA_DIR = "/data/algo/test_data"


class TestBenchmarks(unittest.TestCase):

    def setUp(self):
        self._delete_tmp_dir = True
        self.tmp_dir = tempfile.mkdtemp(dir=TMP_DIR)

    def tearDown(self):
        if self._delete_tmp_dir:
            shutil.rmtree(self.tmp_dir)
        else:
            logging.warning("tmp directory not being deleted, location is: {}".format(
                self.tmp_dir))

    def test_mnist_cnn(self):
        mnist_cnn.main(self.tmp_dir, BENCHMARK_DATA_DIR, num_samples=1)

    def test_resnet_50(self):
        resnet_50.main(self.tmp_dir, BENCHMARK_DATA_DIR, num_samples=1)

    def test_mobilenet_v2(self):
        mobilenet_v2.main(self.tmp_dir, BENCHMARK_DATA_DIR, num_samples=1)

    def test_efficientnet_b0(self):
        efficientnet_b0.main(self.tmp_dir, BENCHMARK_DATA_DIR, num_samples=1)


if __name__ == "__main__":
    unittest.main()
