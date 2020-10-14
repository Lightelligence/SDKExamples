"""
Generic script to run an image classification benchmark
Used for: mnist, resnet50, efficientnet, mobilenet
"""
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
                 output_is_argmax=False):
        # Get full paths
        workload_dir = os.path.join(benchmark_data_dir, workload_dir)
        dataset_dir = os.path.join(workload_dir, dataset_dir)
        self._model_dir = os.path.join(workload_dir, model_dir)
        self._calibration_data_file = os.path.join(dataset_dir, calibration_data_file)
        self._test_data_file = os.path.join(dataset_dir, test_data_file)
        self._test_labels_file = os.path.join(dataset_dir, test_labels_file)

        # Save other params
        self._output_dir = output_dir
        self._input_edge_name = input_edge_name
        self._prediction_edge_name = prediction_edge_name
        self._batch_size = batch_size
        self._k_list = k_list
        self._workload_name = workload_name
        self._output_is_argmax = output_is_argmax

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

    def _get_test_data(self):
        test_data = utils.get_input_data({self._input_edge_name: self._test_data_file},
                                         self._batch_size,
                                         preprocess_fn=self._preprocess_input_data_fn,
                                         num_samples=self._num_samples)

        test_labels = utils.load_np_array(self._test_labels_file,
                                          preprocess_fn=self._preprocess_test_labels_fn)

        return test_data, test_labels

    def _get_calibration_data(self):
        return utils.get_input_data({self._input_edge_name: self._calibration_data_file},
                                    self._batch_size,
                                    preprocess_fn=self._preprocess_input_data_fn,
                                    num_samples=self._num_samples)

    def _get_accuracy(self, batched_inf_out, labels):
        return utils.get_top_k_accuracy(batched_inf_out,
                                        labels,
                                        self._prediction_edge_name,
                                        self._k_list,
                                        output_is_argmax=self._output_is_argmax)

    def run_benchmark(self, graph_type, config):
        """
        Use API functions to run an image classification benchmark
        """

        # Get data
        calibration_data = self._get_calibration_data()
        input_edges = utils.get_input_edges_from_input_data(calibration_data)
        test_data, test_labels = self._get_test_data()

        # Import graph
        original_graph = lt.import_graph(self._model_dir,
                                         config,
                                         graph_type=graph_type,
                                         input_edges=input_edges)

        # Transform graph
        transformed_graph = lt.transform_graph(original_graph,
                                               config,
                                               calibration_data=calibration_data)

        # Run inference on both graphs
        original_outputs = lt.run_functional_simulation(original_graph,
                                                        test_data,
                                                        config)
        transformed_outputs = lt.run_functional_simulation(transformed_graph,
                                                           test_data,
                                                           config)

        # Compare accuracy
        original_accuracy = self._get_accuracy(original_outputs, test_labels)
        transformed_accuracy = self._get_accuracy(transformed_outputs, test_labels)

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
