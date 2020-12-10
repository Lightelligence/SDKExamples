from benchmarks import image_classification_benchmark, utils


def main(output_dir, benchmark_data_dir, num_samples=None, use_full_test_data=False):
    image_classification_benchmark.ImageClassificationBenchmark(
        output_dir=output_dir,
        benchmark_data_dir=benchmark_data_dir,
        workload_dir="mnist_data",
        dataset_dir="datasets",
        model_dir="cnn",
        calibration_data_file="calibration_data.npy",
        test_data_file="test_data.npy",
        test_labels_file="test_labels.npy",
        full_test_data_file_format="test_data_shard_{}.npy",
        full_test_labels_file_format="test_labels_shard_{}.npy",
        input_edge_name="input_example_tensor",
        prediction_edge_name="ArgMax",
        batch_size=250,
        num_samples=num_samples,
        k_list=[1],
        workload_name="mnist_cnn",
        output_is_argmax=True,
        use_full_test_data=use_full_test_data,
    )


if __name__ == "__main__":
    args = utils.parse_args()
    main(args.output_dir,
         args.benchmark_data_dir,
         use_full_test_data=args.use_full_test_data)
