from __future__ import annotations

import csv
import hashlib
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(r"E:\detect uav\record_2k_pure")
OUT_ROOT = ROOT / "all"
SEED = 20260614
TRAIN_RATIO = 0.8
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

SOURCES = [
    ("small", ROOT / "small_fusion_roi640"),
    ("8night", ROOT / "8night"),
    ("ds0607", ROOT / "datasets0607_fusion_clean"),
    ("fusionmix", ROOT / "fusion_roi_datasets_train_val_8_2"),
    ("big", ROOT / "big_fusion"),
]


@dataclass(frozen=True)
class Sample:
    source: str
    source_split: str
    target_split: str | None
    image: Path
    label: Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def label_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect_pair_dir(root: Path, source: str, source_split: str, target_split: str | None) -> tuple[list[Sample], int]:
    image_dir = root / "images"
    label_dir = root / "labels"
    if not image_dir.exists() or not label_dir.exists():
        return [], 0

    samples: list[Sample] = []
    missing_labels = 0
    for image in sorted(image_dir.iterdir()):
        if image.suffix.lower() not in IMAGE_EXTS:
            continue
        label = label_dir / f"{image.stem}.txt"
        if not label.exists():
            missing_labels += 1
            continue
        samples.append(Sample(source, source_split, target_split, image, label))
    return samples, missing_labels


def collect_source(source: str, root: Path) -> tuple[list[Sample], list[Sample], int]:
    if not root.exists():
        raise FileNotFoundError(root)

    assigned: list[Sample] = []
    unsplit: list[Sample] = []
    missing = 0

    flat_samples, flat_missing = collect_pair_dir(root, source, "flat", None)
    unsplit.extend(flat_samples)
    missing += flat_missing

    for split in ("train", "val"):
        split_samples, split_missing = collect_pair_dir(root / split, source, split, split)
        assigned.extend(split_samples)
        missing += split_missing

    eval_samples, eval_missing = collect_pair_dir(root / "eval", source, "eval", None)
    unsplit.extend(eval_samples)
    missing += eval_missing
    return assigned, unsplit, missing


def split_unsplit_samples(samples: list[Sample], seed_offset: int) -> list[Sample]:
    by_source_split: dict[tuple[str, str], list[Sample]] = defaultdict(list)
    for sample in samples:
        by_source_split[(sample.source, sample.source_split)].append(sample)

    out: list[Sample] = []
    for idx, key in enumerate(sorted(by_source_split)):
        group = by_source_split[key][:]
        random.Random(SEED + seed_offset + idx).shuffle(group)
        train_count = round(len(group) * TRAIN_RATIO)
        for sample in group[:train_count]:
            out.append(Sample(sample.source, sample.source_split, "train", sample.image, sample.label))
        for sample in group[train_count:]:
            out.append(Sample(sample.source, sample.source_split, "val", sample.image, sample.label))
    return out


def output_stem(sample: Sample, index: int) -> str:
    return f"{sample.source}_{sample.source_split}_{sample.image.stem}_{index:06d}"


def reset_output() -> None:
    root_resolved = ROOT.resolve()
    out_resolved = OUT_ROOT.resolve()
    if out_resolved.parent != root_resolved or out_resolved.name != "all":
        raise RuntimeError(f"Refusing to reset unexpected output path: {out_resolved}")
    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    for split in ("train", "val"):
        (OUT_ROOT / split / "images").mkdir(parents=True, exist_ok=True)
        (OUT_ROOT / split / "labels").mkdir(parents=True, exist_ok=True)


def write_data_yaml() -> None:
    text = "\n".join(
        [
            f'path: "{OUT_ROOT.as_posix()}"',
            "train: train/images",
            "val: val/images",
            "nc: 1",
            "names: ['uav']",
            "",
        ]
    )
    (OUT_ROOT / "data.yaml").write_text(text, encoding="utf-8")


def read_label_classes(path: Path) -> set[str]:
    classes: set[str] = set()
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.strip().split()
        if parts:
            classes.add(parts[0])
    return classes


def main() -> None:
    assigned: list[Sample] = []
    unsplit: list[Sample] = []
    missing_by_source: Counter[str] = Counter()
    source_roots: dict[str, str] = {}

    for source, root in SOURCES:
        source_roots[source] = str(root)
        src_assigned, src_unsplit, missing = collect_source(source, root)
        assigned.extend(src_assigned)
        unsplit.extend(src_unsplit)
        missing_by_source[source] += missing

    samples = assigned + split_unsplit_samples(unsplit, seed_offset=100)
    reset_output()

    seen_images: dict[str, Sample] = {}
    seen_label_hash: dict[str, str] = {}
    copied_by_source_split: Counter[tuple[str, str]] = Counter()
    copied_by_target_split: Counter[str] = Counter()
    duplicate_by_source: Counter[str] = Counter()
    label_conflict_count = 0
    classes_seen: Counter[str] = Counter()
    duplicates_rows: list[dict[str, str]] = []

    for index, sample in enumerate(samples, start=1):
        image_digest = sha256_file(sample.image)
        label_digest = label_hash(sample.label)
        if image_digest in seen_images:
            duplicate_by_source[sample.source] += 1
            previous = seen_images[image_digest]
            conflict = seen_label_hash[image_digest] != label_digest
            label_conflict_count += int(conflict)
            duplicates_rows.append(
                {
                    "skipped_source": sample.source,
                    "skipped_split": sample.source_split,
                    "skipped_image": str(sample.image),
                    "kept_source": previous.source,
                    "kept_split": previous.source_split,
                    "kept_image": str(previous.image),
                    "label_conflict": "1" if conflict else "0",
                }
            )
            continue

        if sample.target_split not in {"train", "val"}:
            raise RuntimeError(f"Invalid target split for {sample.image}: {sample.target_split}")

        stem = output_stem(sample, index)
        image_out = OUT_ROOT / sample.target_split / "images" / f"{stem}{sample.image.suffix.lower()}"
        label_out = OUT_ROOT / sample.target_split / "labels" / f"{stem}.txt"
        shutil.copy2(sample.image, image_out)
        shutil.copy2(sample.label, label_out)

        seen_images[image_digest] = sample
        seen_label_hash[image_digest] = label_digest
        copied_by_source_split[(sample.source, sample.target_split)] += 1
        copied_by_target_split[sample.target_split] += 1
        for cls in read_label_classes(sample.label):
            classes_seen[cls] += 1

    write_data_yaml()

    duplicates_path = OUT_ROOT / "duplicates_skipped.csv"
    with duplicates_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "skipped_source",
            "skipped_split",
            "skipped_image",
            "kept_source",
            "kept_split",
            "kept_image",
            "label_conflict",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(duplicates_rows)

    report_lines = [
        "all_dataset_merge",
        f"seed={SEED}",
        f"train_ratio_for_flat_or_eval={TRAIN_RATIO}",
        f"output={OUT_ROOT}",
        "",
        "[sources]",
    ]
    for source, root in SOURCES:
        report_lines.append(f"{source}={root}")
    report_lines.extend(["", "[copied_by_source]"])
    for source, _ in SOURCES:
        train = copied_by_source_split[(source, "train")]
        val = copied_by_source_split[(source, "val")]
        report_lines.append(f"{source}: train={train} val={val} total={train + val}")
    report_lines.extend(
        [
            "",
            "[summary]",
            f"train={copied_by_target_split['train']}",
            f"val={copied_by_target_split['val']}",
            f"total={copied_by_target_split['train'] + copied_by_target_split['val']}",
            f"duplicates_skipped={len(duplicates_rows)}",
            f"duplicate_label_conflicts={label_conflict_count}",
            f"missing_label_images={sum(missing_by_source.values())}",
            f"classes_seen={dict(sorted(classes_seen.items()))}",
            "",
            "[missing_label_images_by_source]",
        ]
    )
    for source, _ in SOURCES:
        report_lines.append(f"{source}={missing_by_source[source]}")
    report_lines.extend(["", "[duplicates_skipped_by_source]"])
    for source, _ in SOURCES:
        report_lines.append(f"{source}={duplicate_by_source[source]}")
    report_lines.append("")
    (OUT_ROOT / "merge_report.txt").write_text("\n".join(report_lines), encoding="utf-8")

    print("\n".join(report_lines))


if __name__ == "__main__":
    main()
