# SDKExamples Repo

# Lightelligence SDK Installation and Documentation

Installation instructions, concepts, and SDK documentation can be found on [Read the Docs](https://lightelligence-sdk.readthedocs.io/en/latest/index.html).

# SDKExamples Getting Started

1. Download the Lightelligence SDK according to the [installation instructions](https://lightelligence-sdk.readthedocs.io/en/latest/installation.html#)

2. Clone the SDKExamples repo if you have not done so already and navigate to the root directory of the project.

```sh
git clone https://github.com/Lightelligence/SDKExamples.git
cd SDKExamples
```

3. Install extra dependencies using `pip`. Inside your virtual environment or Docker container

```sh
pip install -r requirements.txt
```

4. Run the setup script to add SDKExamples to your python path

```sh
python setup.py develop
```

# Benchmarks

In this section, we describe how to use our Lightelligence SDK to run certain benchmarks. We recommend looking through the code in the [benchmarks](./benchmarks) directory. The scripts in this folder provide examples of how to use the Lightelligence SDK to run your own custom benchmarks with your own models.

Inside the [benchmarks](./benchmarks) directory, most of the functions use the top level functions defined in the `lt_sdk` module. There are some other more specific functions and packages referenced inside the Lightelligence SDK. The source code for these can be found on the [SDKDocs](https://github.com/Lightelligence/SDKDocs) repository.

In general, the main functions we expect users to work with are found in the `lt_sdk` module. Users may also need to access certain protobuf files, which are all located in the `lt_sdk.proto` folder (all protobuf files can be found [here](https://github.com/Lightelligence/SDKDocs/tree/master/lt_sdk/proto)).

We also recommend using the helper functions in the [benchmarks/utils.py](./benchmarks/utils.py) file for more complicated tasks such as formatting input data and saving protobufs.

## Download Benchmark Data

You can download the graphs and data used for running the benchmarks using the script

```sh
BENCHMARK_DIR="/path/to/benchmark_data_dir"
python benchmarks/download_benchmark_data.py --benchmark_data_dir=$BENCHMARK_DATA_DIR
```

Note that this may take a long time.

## Run a benchmark

We include a set of scripts to run benchmarks in the [benchmarks](./benchmarks) directory. For example, to run the MNIST benchmark, you can use the following command

```sh
OUTPUT_DIR="/path/to/output_dir"
BENCHMARK_DATA_DIR="/path/to/benchmark_data_dir"
python benchmarks/mnist_cnn.py \
--output_dir=$OUTPUT_DIR \
--benchmark_data_dir=$BENCHMARK_DATA_DIR
--use_full_test_data
```

The `--use_full_test_data` flag will use the entire test data set to evaluate the accuracy of the model. Note that this may be very slow when using the full imagenet test data.

## Current Benchmarks

We provide the most up to date benchmarks for certain models in this repository. We plan to add benchmarks for more models in the future. The benchmarks below use the entire MNIST and ImageNet test data sets.

<!--
NOTE:
The values in the accuracy column are filled in during the
upload_to_github.yml pipeline
This pipeline will also create the benchmark/data folder
and populate it with the files referenced in this table
-->

| Workload        | Script                                                           | Accuracy Metric | Original Accuracy              | Transformed Accuracy              | Chrome Tracing                                                                                           | Graphs                                                                                                                                                                                         |
| :-------------- | :--------------------------------------------------------------- | :-------------- | :----------------------------- | :-------------------------------- | -------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| MNIST           | [benchmarks/mnist_cnn.py](./benchmarks/mnist_cnn.py)             | Top 1           | 0.9928       | 0.9919       | [mnist_cnn_transformed.trace](./benchmarks/data/mnist_cnn/mnist_cnn_transformed.trace)                   | [mnist_cnn_original_lgf.pb](./benchmarks/data/mnist_cnn/mnist_cnn_original_lgf.pb), [mnist_cnn_transformed_lgf.pb](./benchmarks/data/mnist_cnn/mnist_cnn_transformed_lgf.pb)                   |
| Resnet 50       | [benchmarks/resnet_50.py](./benchmarks/resnet_50.py)             | Top 5           | 0.9306       | 0.9299       | [resnet_50_transformed.trace](./benchmarks/data/resnet_50/resnet_50_transformed.trace)                   | [resnet_50_original_lgf.pb](./benchmarks/data/resnet_50/resnet_50_original_lgf.pb), [resnet_50_transformed_lgf.pb](./benchmarks/data/resnet_50/resnet_50_transformed_lgf.pb)                   |
| Mobilenet V2    | [benchmarks/mobilenet_v2.py](./benchmarks/mobilenet_v2.py)       | Top 5           | 0.7522    | 0.7081    | [mobilenet_v2_transformed.trace](./benchmarks/data/mobilenet_v2/mobilenet_v2_transformed.trace)          | [mobilenet_v2_original_lgf.pb](./benchmarks/data/mobilenet_v2/mobilenet_v2_original_lgf.pb), [mobilenet_v2_transformed_lgf.pb](./benchmarks/data/mobilenet_v2/mobilenet_v2_transformed_lgf.pb) |
| Efficientnet B0 | [benchmarks/efficientnet_b0.py](./benchmarks/efficientnet_b0.py) | Top 5           | 0.9303 | 0.8921 | [efficientnet_b0_transformed.trace](./benchmarks/data/efficientnet_b0/efficientnet_b0_transformed.trace) | [efficientnet_b0_original_lgf.pb](./benchmarks/data/efficientnet_b0_original_lgf.pb), [efficientnet_b0_transformed_lgf.pb](./benchmarks/efficientnet_b0/efficientnet_b0_transformed_lgf.pb)    |
