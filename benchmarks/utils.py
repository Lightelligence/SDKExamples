import argparse
import json
import logging
import os
import shutil
import tarfile

import numpy as np
from azure.storage import blob
from lt_sdk.data import batch, named_tensor
from lt_sdk.graph import full_graph_pipeline
from lt_sdk.graph.transform_graph import utils as lt_utils
from lt_sdk.proto import dtypes_pb2
from lt_sdk.visuals import sim_result_to_trace

REQUIRED_SUBDIRS = {
    "mnist_data",
    "imagenet_data",
}


def check_benchmark_data_dir(benchmark_data_dir):
    if not os.path.exists(benchmark_data_dir):
        raise ValueError("Benchmark data directory does not exist")

    subdirs = {
        d for d in os.listdir(benchmark_data_dir)
        if os.path.isdir(os.path.join(benchmark_data_dir,
                                      d))
    }

    if not REQUIRED_SUBDIRS.issubset(subdirs):
        raise ValueError("Did not find expected subdirectories: {}".format(
            REQUIRED_SUBDIRS.difference(subdirs)))


def download_benchmark_data(benchmark_data_dir, blob_url):
    if os.path.exists(benchmark_data_dir):
        raise ValueError("Data directory already exists")
    else:
        os.makedirs(benchmark_data_dir)

    blob_client = blob.BlobClient.from_blob_url(blob_url=blob_url)

    fname = os.path.join(benchmark_data_dir, "benchmark_data.tar.gz")
    with open(fname, "wb") as f:
        f.write(blob_client.download_blob().readall())

    with tarfile.open(fname) as tar:
        tar.extractall(path=benchmark_data_dir)

    check_benchmark_data_dir(benchmark_data_dir)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        help="directory to save output data",
    )
    parser.add_argument("--benchmark_data_dir",
                        type=str,
                        help=("directory where the benchmark data is stored"))
    args = parser.parse_args()

    check_benchmark_data_dir(args.benchmark_data_dir)
    return args


def get_top_k_predictions(predictions, k, output_is_argmax=False):
    """Returns the top k predictions from numpy array predictions"""
    if output_is_argmax:
        assert (k == 1)
        assert (predictions.shape[1] == 1)
        return predictions
    else:
        assert (predictions.shape[1] >= k)
        return predictions.argsort(axis=1)[:, -k:]


def get_top_k_accuracy(batched_inf_out,
                       labels,
                       prediction_edge,
                       k_list,
                       output_is_argmax=False):
    # Keep counters for number of correct predictions and total number of predictions
    correct = {k: 0 for k in k_list}
    total = 0

    # Get the predictions
    predictions = []
    for inf_out in batched_inf_out.batches:
        for named_ten in inf_out.results:
            if named_ten.edge_info.name.startswith(prediction_edge):
                predictions.append(
                    lt_utils.tensor_pb_to_array(named_ten.data,
                                                np.float32))
    predictions = np.concatenate(predictions, axis=0)

    # Format the arrays
    num_samples = min(labels.shape[0], predictions.shape[0])
    predictions = predictions[:num_samples]
    predictions = predictions.reshape(num_samples, -1)
    labels = labels[:num_samples]
    labels = labels.reshape(num_samples, 1)

    # Calculate top k accuracy for each value of k
    total += num_samples
    for k in k_list:
        top_k = get_top_k_predictions(predictions, k, output_is_argmax=output_is_argmax)
        correct[k] += np.sum(labels == top_k)

    return {k: correct[k] / total for k in k_list}


def save_protobuf(protobuf, dirname, fname):
    with open(os.path.join(dirname, fname), "wb") as f:
        f.write(protobuf.SerializeToString())


def read_protobuf(proto_cls, dirname, fname):
    protobuf = proto_cls()
    with open(os.path.join(dirname, fname), "rb") as f:
        protobuf.ParseFromString(f.read())

    return protobuf


def save_dict(py_dict, dirname, fname):
    with open(os.path.join(dirname, fname), "w") as f:
        json.dump(py_dict, f)


def read_dict(dirname, fname):
    with open(os.path.join(dirname, fname), "r") as f:
        py_dict = json.load(f)

    return py_dict


def save_outputs(original_graph,
                 original_outputs,
                 original_accuracy,
                 transformed_graph,
                 transformed_outputs,
                 transformed_accuracy,
                 config,
                 exec_stats,
                 output_dir,
                 clear_dir=True,
                 fname_prefix=""):
    # Clean directory if it already exists
    if os.path.exists(output_dir):
        if clear_dir:
            shutil.rmtree(output_dir)
            os.makedirs(output_dir)
    else:
        os.makedirs(output_dir)

    # Get prefix
    if fname_prefix != "" and not fname_prefix.endswith("_"):
        fname_prefix += "_"

    # Save original info
    save_protobuf(original_graph.as_lgf_pb(),
                  output_dir,
                  fname_prefix + "original_lgf.pb")
    save_protobuf(original_outputs,
                  output_dir,
                  fname_prefix + "original_batched_inf_out.pb")
    save_dict(original_accuracy, output_dir, fname_prefix + "original_accuracy.json")

    # Save transformed info
    save_protobuf(transformed_graph.as_lgf_pb(),
                  output_dir,
                  fname_prefix + "transformed_lgf.pb")
    save_protobuf(transformed_outputs,
                  output_dir,
                  fname_prefix + "transformed_batched_inf_out.pb")
    save_dict(transformed_accuracy,
              output_dir,
              fname_prefix + "transformed_accuracy.json")
    save_protobuf(config.to_proto(), output_dir, fname_prefix + "transformed_config.pb")
    save_protobuf(exec_stats, output_dir, fname_prefix + "transformed_exec_stats.pb")
    sim_result_to_trace.instruction_trace(
        os.path.join(output_dir,
                     fname_prefix + "transformed.trace"),
        exec_stats,
        config.hw_specs,
        config.sim_params)


def load_np_array(path, preprocess_fn=None):
    array = np.load(path)
    if preprocess_fn is not None:
        array = preprocess_fn(array)

    return array


def get_input_data(filenames,
                   batch_size,
                   preprocess_fn=None,
                   num_samples=None,
                   dtype=dtypes_pb2.DT_FLOAT):
    names = []
    tensors = []
    for name, path in filenames.items():
        names.append(name)
        tensors.append(load_np_array(path, preprocess_fn=preprocess_fn))

    named_tensors = named_tensor.NamedTensorSet(names, tensors, dtype=dtype)

    if num_samples is not None:

        def get_subset(array):
            return array[:num_samples]

        named_tensors.apply_all(get_subset)

        if num_samples < batch_size:
            logging.warning(
                "Num samples {} is less than batch size {}. Consider reducing batch size"
                .format(num_samples,
                        batch_size))

    batched_inputs = batch.batch_inputs(named_tensors, batch_size=batch_size)

    return batched_inputs


def get_input_edges_from_input_data(input_data):
    return full_graph_pipeline.extract_edge_from_data(input_data)
