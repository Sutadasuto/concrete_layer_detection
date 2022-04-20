import argparse
import os

import data

from distutils.util import strtobool

from additional_methods import Bilateral_Filtered_Canny


def main(args):
    results_dir = "results_additional_methods"
    results_train_dir = os.path.join(results_dir, "results_training")
    results_train_min_loss_dir = results_train_dir + "_min_val_loss"
    results_validation_dir = os.path.join(results_dir, "results_validation")
    results_validation_min_loss_dir = results_validation_dir + "_min_val_loss"

    # Define the testing image paths
    paths = data.create_image_paths(args.images_path, args.gt_path)
    # Ensure that ground truths are binary and uint8
    data.annotation_thresholding(paths[1, :])

    method_1 = Bilateral_Filtered_Canny({'d': 30, 'sigmaColor': 100, 'sigmaSpace': 80}, {'threshold1': 15, 'threshold2': 15})
    data.evaluate_model_on_paths(method_1, paths, results_validation_dir, args)


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-ip", "--images_path", type=str,
                        help="Path to the directory containing the images.")
    parser.add_argument("-gt", "--gt_path", type=str, default=None,
                        help="Path to a directory containing ground truth annotations for evaluation. If 'None', "
                             "no evaluation is performed.")
    parser.add_argument("--save_to", type=str, default=None,
                        help="Results will be saved in this location. If not provided, a folder 'results_date_hour'"
                             "will be created for this purpose.")

    args_dict = parser.parse_args(args)
    for attribute in args_dict.__dict__.keys():
        if args_dict.__getattribute__(attribute) == "None":
            args_dict.__setattr__(attribute, None)
        if args_dict.__getattribute__(attribute) == "True" or args_dict.__getattribute__(attribute) == "False":
            args_dict.__setattr__(attribute, bool(strtobool(args_dict.__getattribute__(attribute))))
    return args_dict


if __name__ == "__main__":
    args = parse_args()
    main(args)