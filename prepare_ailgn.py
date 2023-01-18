import argparse
import yaml

from preprocessor import complex, heather, multi, korean
from audio.db_normalization import amplitude_normalize


def main(config):
    if "complex" in config["dataset"]:
        complex.prepare_align(config)
        amplitude_normalize(config)
    if "Heather" in config["dataset"]:
        heather.prepare_align(config)
        amplitude_normalize(config)
    if "multi" in config["dataset"]:
        multi.prepare_align(config)
        amplitude_normalize(config)
    if "korean" in config["dataset"]:
        korean.prepare_align(config)
        # amplitude_normalize(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./config/korean/preprocess.yaml", type=str, help="path to preprocess.yaml")
    args = parser.parse_args()

    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    main(config)
