import argparse


def main():
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--allocated_hosts",
        type=str,
        required=True,
        help="Allocated host nodes (no default)",
    )
    parser.add_argument(
        "--current_host",
        type=str,
        required=True,
        help="Current host node (no default)",
    )
    args = parser.parse_args()

    print(f"Allocated: {args.allocated_hosts}")
    print(f"Current: {args.current_host}")


if __name__=="__main__":
    main()