"""
Generic script to run an image classification benchmark
Used for: mnist, resnet50, efficientnet, mobilenet
"""
import glob
import os

import lt_sdk as lt
from lt_sdk.proto import graph_types_pb2, hardware_configs_pb2

from benchmarks import utils


class ImageClassificationBenchmark:

    def __init__(self,
                 output_dir,
                 benchmark_data_dir,
                 workload_dir,
                 dataset_dir,
                 model_dir,
                 calibration_data_file,
                 test_data_file,
                 test_labels_file,
                 full_test_data_file_format,
                 full_test_labels_file_format,
                 input_edge_name,
                 prediction_edge_name,
                 batch_size,
                 num_samples=None,
                 hw_cfg=None,
                 graph_type=None,
                 config=None,
                 preprocess_input_data_fn=None,
                 preprocess_test_labels_fn=None,
                 k_list=[1],
                 workload_name="",
                 output_is_argmax=False,
                 use_full_test_data=False):
        # Get full paths
        workload_dir = os.path.join(benchmark_data_dir, workload_dir)
        dataset_dir = os.path.join(workload_dir, dataset_dir)
        self._model_dir = os.path.join(workload_dir, model_dir)
        self._calibration_data_file = os.path.join(dataset_dir, calibration_data_file)
        self._test_data_file = os.path.join(dataset_dir, test_data_file)
        self._test_labels_file = os.path.join(dataset_dir, test_labels_file)
        self._full_test_data_file_format = os.path.join(dataset_dir,
                                                        full_test_data_file_format)
        self._full_test_labels_file_format = os.path.join(dataset_dir,
                                                          full_test_labels_file_format)

        # Save other params
        self._output_dir = output_dir
        self._input_edge_name = input_edge_name
        self._prediction_edge_name = prediction_edge_name
        self._batch_size = batch_size
        self._k_list = k_list
        self._workload_name = workload_name
        self._output_is_argmax = output_is_argmax
        self._use_full_test_data = use_full_test_data

        # Make things more efficient by decreasing batch size to match
        # num samples if possible
        self._num_samples = num_samples
        if self._num_samples is not None and self._num_samples < self._batch_size:
            self._batch_size = num_samples

        self._preprocess_input_data_fn = preprocess_input_data_fn
        self._preprocess_test_labels_fn = preprocess_test_labels_fn

        # Default values
        hw_cfg = hw_cfg or hardware_configs_pb2.DELTA
        graph_type = graph_type or graph_types_pb2.TFSavedModel
        config = config or lt.get_default_config(hw_cfg=hw_cfg, graph_type=graph_type)

        # Run the benchmark
        self.run_benchmark(graph_type, config)

    def _get_test_data_from_file(self, test_data_file, test_labels_file):
        test_data = utils.get_input_data({self._input_edge_name: test_data_file},
                                         self._batch_size,
                                         preprocess_fn=self._preprocess_input_data_fn,
                                         num_samples=self._num_samples)

        test_labels = utils.load_np_array(test_labels_file,
                                          preprocess_fn=self._preprocess_test_labels_fn)

        return test_data, test_labels

    def _sorted_paths(self, file_format):
        return sorted(glob.glob(file_format.format("*")))

    def _num_test_shards(self):
        if self._use_full_test_data:
            return len(self._sorted_paths(self._full_test_data_file_format))
        else:
            return 1

    def _get_test_data_shard(self, i):
        if self._use_full_test_data:
            return self._get_test_data_from_file(
                self._sorted_paths(self._full_test_data_file_format)[i],
                self._sorted_paths(self._full_test_labels_file_format)[i])
        else:
            return self._get_test_data_from_file(self._test_data_file,
                                                 self._test_labels_file)

    def _get_calibration_data(self):
        return utils.get_input_data({self._input_edge_name: self._calibration_data_file},
                                    self._batch_size,
                                    preprocess_fn=self._preprocess_input_data_fn,
                                    num_samples=self._num_samples)

    def _run_inference(self, graph, config, test_data, test_labels):
        # Run inference through functional simulator
        batched_inf_out = lt.run_functional_simulation(graph, test_data, config)

        # Get top k accuracy info
        correct, total = utils.get_top_k_accuracy(
            batched_inf_out,
            test_labels,
            self._prediction_edge_name,
            self._k_list,
            output_is_argmax=self._output_is_argmax)

        return batched_inf_out, correct, total

    def run_benchmark(self, graph_type, config):
        """
        Use API functions to run an image classification benchmark
        """

        # Get data
        calibration_data = self._get_calibration_data()
        input_edges = utils.get_input_edges_from_input_data(calibration_data)

        # Import graph
        original_graph = lt.import_graph(self._model_dir,
                                         config,
                                         graph_type=graph_type,
                                         input_edges=input_edges)

        # Transform graph
        transformed_graph = lt.transform_graph(original_graph,
                                               config,
                                               calibration_data=calibration_data)

        # Keep track of correct and total for both graphs
        original_correct = {k: 0 for k in self._k_list}
        transformed_correct = {k: 0 for k in self._k_list}
        total = 0
        original_outputs = []
        transformed_outputs = []

        for i in range(self._num_test_shards()):
            # Run inference on both graphs
            test_data, test_labels = self._get_test_data_shard(i)
            original_shard_outputs, original_shard_correct, shard_total = \
                self._run_inference(original_graph, config, test_data, test_labels)
            transformed_shard_outputs, transformed_shard_correct, _ = \
                self._run_inference(transformed_graph, config, test_data, test_labels)

            # Update correct and total
            for k in self._k_list:
                original_correct[k] += original_shard_correct[k]
                transformed_correct[k] += transformed_shard_correct[k]
            total += shard_total

            # Store the outputs
            original_outputs.append(original_shard_outputs)
            transformed_outputs.append(transformed_shard_outputs)

        # Compute the accuracy
        original_accuracy = {k: original_correct[k] / total for k in self._k_list}
        transformed_accuracy = {k: transformed_correct[k] / total for k in self._k_list}

        print("Original accuracy: {}".format(original_accuracy))
        print("Transformed accuracy: {}".format(transformed_accuracy))

        # Run performance simulation
        exec_stats = lt.run_performance_simulation(transformed_graph, config)

        # Save everything
        utils.save_outputs(original_graph,
                           original_outputs,
                           original_accuracy,
                           transformed_graph,
                           transformed_outputs,
                           transformed_accuracy,
                           config,
                           exec_stats,
                           self._output_dir,
                           fname_prefix=self._workload_name)
