"""Export stratified training and test CSVs for the maintenance project."""

import argparse
from pathlib import Path

from sklearn.model_selection import train_test_split

from .features import ReadingFeatures
from .ingest import DEFAULT_INPUT, TARGET, load_readings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=Path("data/prepared"))
    parser.add_argument("--feature-set", choices=("sensor", "all"), default="sensor")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    readings = load_readings(args.input)
    train, test = train_test_split(
        readings,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=readings[TARGET],
    )
    transformer = ReadingFeatures(args.feature_set).fit(train)
    train_features = transformer.transform(train).assign(**{TARGET: train[TARGET]})
    test_features = transformer.transform(test).assign(**{TARGET: test[TARGET]})

    output_path = args.output_dir / args.feature_set
    outputs = (
        (train_features, output_path / "train.csv"),
        (test_features, output_path / "test.csv"),
    )
    for _, path in outputs:
        if path.resolve() == args.input.resolve():
            parser.error("An output path points to the input CSV.")
    output_path.mkdir(parents=True, exist_ok=True)
    for features, path in outputs:
        features.to_csv(path, index=False)
        print(f"Wrote {len(features):,} rows to {path}")


if __name__ == "__main__":
    main()
