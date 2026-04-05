from pathlib import Path

def split_data(tweet_raw_path: str, emoji_raw_path: str, output_folder: str, split_ratio: float = 0.9):
    
    # Verify output folder exists
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    Path(output_folder).joinpath('tweet_test.txt').touch(exist_ok=True)
    Path(output_folder).joinpath('tweet_train.txt').touch(exist_ok=True)
    Path(output_folder).joinpath('emoji_test.txt').touch(exist_ok=True)
    Path(output_folder).joinpath('emoji_train.txt').touch(exist_ok=True)

    # Verify raw files exist
    if not Path(tweet_raw_path).exists():
        raise FileNotFoundError(f"File not found: {tweet_raw_path}")
    
    if not Path(emoji_raw_path).exists():
        raise FileNotFoundError(f"File not found: {emoji_raw_path}")

    # Read and split tweet / emoji raw files.
    with open(Path(tweet_raw_path), 'r', encoding='utf-8') as f:
        lines = f.readlines()

    with open(Path(output_folder) / Path('tweet_test.txt'), 'w', encoding='utf-8') as f:
        f.writelines(lines[int(len(lines) * split_ratio):])

    with open(Path(output_folder) / Path('tweet_train.txt'), 'w', encoding='utf-8') as f:
        f.writelines(lines[:int(len(lines) * split_ratio)])

    with open(Path(emoji_raw_path), 'r', encoding='utf-8') as f:
        lines = f.readlines()

    with open(Path(output_folder) / Path('emoji_test.txt'), 'w', encoding='utf-8') as f:
        f.writelines(lines[int(len(lines) * split_ratio):])

    with open(Path(output_folder) / Path('emoji_train.txt'), 'w', encoding='utf-8') as f:
        f.writelines(lines[:int(len(lines) * split_ratio)])



if __name__ == "__main__":

    tweets_raw_file_path = Path(__file__).parent.parent / Path("data/tweets.txt")
    emoji_raw_file_path = Path(__file__).parent.parent / Path("data/emoji.txt")
    output_folder = Path(__file__).parent.parent / Path("data/split")

    split_data(
        tweet_raw_path=tweets_raw_file_path,
        emoji_raw_path=emoji_raw_file_path,
        output_folder=output_folder,
        split_ratio=0.9
    )