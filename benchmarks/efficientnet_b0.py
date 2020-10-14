import numpy as np

from benchmarks import image_classification_benchmark, utils

STDDEV_RGB = np.array([0.229 * 255, 0.224 * 255, 0.225 * 255])


def preprocess_input_data(array):
    return array.astype(np.float32) / STDDEV_RGB


def preprocess_test_labels(array):
    return array - 1


def main(output_dir, benchmark_data_dir, num_samples=None):
    image_classification_benchmark.ImageClassificationBenchmark(
        output_dir=output_dir,
        benchmark_data_dir=benchmark_data_dir,
        workload_dir="imagenet_data",
        dataset_dir="datasets",
        model_dir="efficientnet_b0",
        calibration_data_file="calibration_data.npy",
        test_data_file="test_data.npy",
        test_labels_file="test_labels.npy",
        input_edge_name="images",
        prediction_edge_name="Softmax",
        batch_size=256,
        num_samples=num_samples,
        preprocess_input_data_fn=preprocess_input_data,
        preprocess_test_labels_fn=preprocess_test_labels,
        k_list=[5],
        workload_name="efficientnet_b0",
    )


if __name__ == "__main__":
    args = utils.parse_args()
    main(args.output_dir, args.benchmark_data_dir)
