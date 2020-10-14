import numpy as np

from benchmarks import image_classification_benchmark, utils


def preprocess_input_data(array):
    return array.astype(np.float32) / 128 - 1


def main(output_dir, benchmark_data_dir, num_samples=None):
    image_classification_benchmark.ImageClassificationBenchmark(
        output_dir=output_dir,
        benchmark_data_dir=benchmark_data_dir,
        workload_dir="imagenet_data",
        dataset_dir="datasets",
        model_dir="mobilenet_v2",
        calibration_data_file="calibration_data.npy",
        test_data_file="test_data.npy",
        test_labels_file="test_labels.npy",
        input_edge_name="input",
        prediction_edge_name="MobilenetV2/Predictions/Reshape_1",
        batch_size=256,
        num_samples=num_samples,
        preprocess_input_data_fn=preprocess_input_data,
        k_list=[5],
        workload_name="mobilenet_v2",
    )


if __name__ == "__main__":
    args = utils.parse_args()
    main(args.output_dir, args.benchmark_data_dir)
