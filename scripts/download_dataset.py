from __future__ import annotations

from pathlib import Path


def main() -> None:
    datasets_dir = Path("datasets")
    datasets_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created datasets directory at: {datasets_dir.resolve()}")
    print("Download logic will be added in the evaluation step.")


if __name__ == "__main__":
    main()