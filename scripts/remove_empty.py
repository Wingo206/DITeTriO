import os
import argparse

def remove_empty_files(directory):
    """
    Recursively removes empty files from the given directory.

    Args:
        directory (str): Path to the directory to clean up.

    Returns:
        int: Count of empty files removed.
    """
    empty_files_removed = 0

    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if os.path.isfile(file_path) and os.path.getsize(file_path) == 0:
                os.remove(file_path)
                print(f"Removed empty file: {file_path}")
                empty_files_removed += 1

    print(f"Total empty files removed: {empty_files_removed}")
    return empty_files_removed

def main():
    parser = argparse.ArgumentParser(description="Remove all empty files from a directory recursively.")
    parser.add_argument(
        "--directory",
        type=str,
        help="Path to the directory to clean up."
    )

    args = parser.parse_args()
    dir_path = args.directory

    if os.path.isdir(dir_path):
        removed_count = remove_empty_files(dir_path)
        print(f"Clean-up complete. {removed_count} empty files removed.")
    else:
        print("Invalid directory path provided.")

if __name__ == "__main__":
    main()

