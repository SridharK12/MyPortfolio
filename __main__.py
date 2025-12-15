import argparse
from .train import train


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to input diabetes CSV file"
    )

    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Directory to save the trained model"
    )

    args = parser.parse_args()

    train(
        data_path=args.data_path,
        model_dir=args.model_dir
    )


if __name__ == "__main__":
    main()
