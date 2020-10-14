import logging
import os
import subprocess
import traceback

from benchmarks import efficientnet_b0, mnist_cnn, mobilenet_v2, resnet_50, utils

BENCHMARK_MODULES = {
    efficientnet_b0: "5",
    mnist_cnn: "1",
    mobilenet_v2: "5",
    resnet_50: "5",
}

README = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "README.md")

TRANSFORMED_JSON_SUFFIX = "_transformed_accuracy.json"
ORIGINAL_JSON_SUFFIX = "_original_accuracy.json"
TRANSFORMED_MATCH_SUFFIX = "_TRANSFORMED_ACC"
ORIGINAL_MATCH_SUFFIX = "_ORIGINAL_ACC"


def update_readme(output_dir, module_name, acc_key, json_suffix, match_suffix):
    acc_dict = utils.read_dict(output_dir, module_name + json_suffix)
    acc_val = acc_dict[acc_key]
    acc_match = "({})".format(module_name.upper() + match_suffix.upper())
    subprocess.call(
        ["sed",
         "-i",
         "-e",
         "s/{}/{:.4f}/g".format(acc_match,
                                acc_val),
         README])


def run_benchmark(output_dir, benchmark_data_dir, module, module_name, acc_key):
    output_dir = os.path.join(output_dir, module_name)
    module.main(output_dir, benchmark_data_dir)

    update_readme(output_dir,
                  module_name,
                  acc_key,
                  ORIGINAL_JSON_SUFFIX,
                  ORIGINAL_MATCH_SUFFIX)
    update_readme(output_dir,
                  module_name,
                  acc_key,
                  TRANSFORMED_JSON_SUFFIX,
                  TRANSFORMED_MATCH_SUFFIX)


def main(output_dir, benchmark_data_dir):
    for module, acc_key in BENCHMARK_MODULES.items():
        try:
            module_name = module.__name__.split(".")[-1]
            run_benchmark(output_dir, benchmark_data_dir, module, module_name, acc_key)
        except Exception:
            logging.error(traceback.format_exc())
            logging.error(
                "Benchmark {} failed, continuing to next benchmark".format(module_name))


if __name__ == "__main__":
    args = utils.parse_args()
    main(args.output_dir, args.benchmark_data_dir)
