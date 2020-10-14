import argparse

from benchmarks import utils

BLOB_URL = "https://swtrainingdata.blob.core.windows.net/sdkexamples/benchmark_data"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark_data_dir",
        type=str,
        help=("directory where the saved model, " +
              "calibration, and test data are stored"),
    )
    args = parser.parse_args()
    utils.download_benchmark_data(args.benchmark_data_dir, BLOB_URL)
