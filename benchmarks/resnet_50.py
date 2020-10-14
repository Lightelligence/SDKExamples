from benchmarks import image_classification_benchmark, utils


def main(output_dir, benchmark_data_dir, num_samples=None):
    image_classification_benchmark.ImageClassificationBenchmark(
        output_dir=output_dir,
        benchmark_data_dir=benchmark_data_dir,
        workload_dir="imagenet_data",
        dataset_dir="datasets",
        model_dir="resnet_AWS_NHWC",
        calibration_data_file="calibration_data.npy",
        test_data_file="test_data.npy",
        test_labels_file="test_labels.npy",
        input_edge_name="input_example_tensor",
        prediction_edge_name="softmax_tensor",
        batch_size=640,
        num_samples=num_samples,
        k_list=[5],
        workload_name="resnet_50",
    )


if __name__ == "__main__":
    args = utils.parse_args()
    main(args.output_dir, args.benchmark_data_dir)
