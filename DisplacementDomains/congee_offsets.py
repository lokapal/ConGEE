#!/usr/bin/env python3
"""Prepare a ConGEE TAD displacement sweep and collect its eight-column results.

Python 3.8+; standard library only. Run with --help, prepare --help, collect --help.
This program does not invoke or modify do_lists.sh, ConGEE, or cells.list.
"""

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import re
import shutil
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


VERSION = "1.1.1"
MISSING = {"na", "n/a", "nan", "none", "null", ".", "-"}
MANIFEST_FIELDS = [
    "shifted_label", "map", "offset_bp", "offset_kb", "bed_path",
    "source_bed", "original_domains", "retained_domains", "excluded_domains",
    "source_out_of_bounds_domains", "shift_edge_excluded_domains",
]
RESULT_FIELDS = [
    "dataset", "map", "offset_bp", "offset_kb", "z_e", "p_z_e", "z_c",
    "p_z_c_left", "dispersion", "concordance", "status", "shifted_map",
]


class InputError(Exception):
    """An actionable input or configuration error."""


def fail(message):
    raise InputError(message)


def resolved(value):
    return Path(value).expanduser().resolve()


def read_lines(path):
    opener = gzip.open if path.name.lower().endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8-sig") as handle:
        yield from enumerate(handle, 1)


def hash_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_sizes(path):
    sizes = {}
    for line_no, raw in read_lines(path):
        text = raw.strip()
        if not text or text.startswith("#"):
            continue
        fields = text.split()
        if len(fields) != 2:
            fail("{}:{}: expected chromosome and length.".format(path, line_no))
        chrom, value = fields
        try:
            length = int(value)
        except ValueError:
            fail("{}:{}: chromosome length is not an integer.".format(path, line_no))
        if length <= 0 or chrom in sizes:
            fail("{}:{}: nonpositive length or duplicate chromosome {}.".format(
                path, line_no, chrom))
        sizes[chrom] = length
    if not sizes:
        fail("No chromosome sizes found in {}.".format(path))
    return sizes


def read_map_list(path):
    entries, paths, labels = [], set(), set()
    for line_no, raw in read_lines(path):
        text = raw.strip()
        if not text or text.startswith("#"):
            continue
        source = Path(text).expanduser()
        if not source.is_absolute():
            source = path.parent / source
        # Keep the listed basename and compression suffix for symlinked inputs.
        # Resolve separately to detect two aliases of the same physical file.
        source = Path(os.path.abspath(source))
        canonical = source.resolve()
        if not source.is_file():
            fail("{}:{}: BED file does not exist: {}".format(path, line_no, source))
        name = source.name
        if name.lower().endswith(".bed.gz"):
            label = name[:-7]
        elif name.lower().endswith(".bed"):
            label = name[:-4]
        else:
            fail("{}:{}: expected a .bed or .bed.gz filename.".format(path, line_no))
        if not label or not re.fullmatch(r"[A-Za-z0-9_.-]+", label):
            fail("Map basename must contain only letters, digits, _, . or -: {}".format(name))
        if canonical in paths or label in labels:
            fail("Duplicate BED path or ambiguous map basename: {}".format(source))
        paths.add(canonical)
        labels.add(label)
        entries.append((label, source))
    if not entries:
        fail("No BED paths found in {}.".format(path))
    return entries


def check_bed12(fields, start, end, location):
    if len(fields) != 12:
        fail("{}: --bed-format bed12 requires exactly 12 fields.".format(location))
    try:
        thick_start, thick_end = int(fields[6]), int(fields[7])
        block_count = int(fields[9])
        block_sizes = [int(x) for x in fields[10].rstrip(",").split(",")]
        block_starts = [int(x) for x in fields[11].rstrip(",").split(",")]
    except ValueError:
        fail("{}: malformed BED12 coordinate/block fields.".format(location))
    if not start <= thick_start <= thick_end <= end:
        fail("{}: BED12 thick coordinates are outside the interval.".format(location))
    if block_count <= 0 or len(block_sizes) != block_count or len(block_starts) != block_count:
        fail("{}: BED12 blockCount does not match its blocks.".format(location))
    previous_end = 0
    for relative_start, length in zip(block_starts, block_sizes):
        if relative_start < previous_end or length <= 0 or relative_start + length > end - start:
            fail("{}: invalid or overlapping BED12 blocks.".format(location))
        previous_end = relative_start + length
    if block_starts[0] != 0 or previous_end != end - start:
        fail("{}: BED12 blocks do not span the outer interval.".format(location))


def read_bed(path, sizes, mode, out_of_bounds="error"):
    records = []
    columns = None
    for line_no, raw in read_lines(path):
        text = raw.rstrip("\r\n")
        clean = text.strip()
        if (not clean or clean.startswith("#") or clean == "track" or
                clean.startswith(("track ", "track\t", "browser ", "browser\t"))):
            records.append((line_no, text, None))
            continue
        fields = text.split("\t") if "\t" in text else clean.split()
        if len(fields) < 3:
            fail("{}:{}: fewer than three BED fields.".format(path, line_no))
        if columns is None:
            columns = len(fields)
        elif columns != len(fields):
            fail("{}:{}: inconsistent BED column count.".format(path, line_no))
        if mode == "auto" and len(fields) > 6:
            fail("{}:{}: more than six fields. Select --bed-format interval for "
                 "opaque extra annotations, or --bed-format bed12 for standard BED12.".format(
                     path, line_no))
        try:
            start, end = int(fields[1]), int(fields[2])
        except ValueError:
            fail("{}:{}: BED start/end must be integers.".format(path, line_no))
        chrom = fields[0]
        if chrom not in sizes:
            fail("{}:{}: chromosome {} is absent from the sizes file.".format(path, line_no, chrom))
        if start >= end:
            fail("{}:{}: invalid original interval {}:{}-{}; start must be less than end.".format(
                path, line_no, chrom, start, end))
        if (start < 0 or end > sizes[chrom]) and out_of_bounds == "error":
            fail("{}:{}: invalid original interval {}:{}-{} (length {}).".format(
                path, line_no, chrom, start, end, sizes[chrom]) +
                 " To exclude out-of-bounds source intervals from every offset, "
                 "rerun with --out-of-bounds exclude. No coordinates will be clipped.")
        if mode == "bed12":
            check_bed12(fields, start, end, "{}:{}".format(path, line_no))
        records.append((line_no, None, fields))
    if columns is None:
        fail("No intervals found in {}.".format(path))
    return records


def offset_tag(offset):
    if offset == 0:
        return "offset_0bp"
    return "offset_{}{:06d}bp".format("m" if offset < 0 else "p", abs(offset))


def write_table(path, fields, rows):
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def prepare(args):
    tad_list, sizes_path, output = map(resolved, (args.tads_list, args.chrom_sizes, args.out_dir))
    if output.exists():
        fail("Output directory already exists; use a new --out-dir: {}".format(output))
    if args.step_kb <= 0 or args.stop_kb < args.start_kb:
        fail("Require a positive step and stop >= start.")
    if (args.stop_kb - args.start_kb) % args.step_kb:
        fail("The step must divide the start-to-stop span exactly.")
    offsets = sorted(set(range(args.start_kb * 1000, args.stop_kb * 1000 + 1,
                               args.step_kb * 1000)) | {0})
    sizes = read_sizes(sizes_path)
    maps = read_map_list(tad_list)
    prepared, excluded, counts = [], [], []
    for label, source in maps:
        records = read_bed(source, sizes, args.bed_format, args.out_of_bounds)
        kept, total_by_chr, kept_by_chr = set(), Counter(), Counter()
        source_bad_by_chr = Counter()
        for line_no, header, fields in records:
            if fields is None:
                continue
            chrom, start, end = fields[0], int(fields[1]), int(fields[2])
            total_by_chr[chrom] += 1
            left = start + offsets[0] < 0
            right = end + offsets[-1] > sizes[chrom]
            source_bad = start < 0 or end > sizes[chrom]
            if source_bad:
                source_bad_by_chr[chrom] += 1
            if left or right:
                excluded.append({"map": label, "source_bed": str(source), "source_line": line_no,
                                 "chrom": chrom, "start": start, "end": end,
                                 "chrom_length": sizes[chrom],
                                 "source_left_overhang_bp": max(0, -start),
                                 "source_right_overhang_bp": max(0, end - sizes[chrom]),
                                 "reason": "source_out_of_bounds" if source_bad else
                                 ("both_edges" if left and right else
                                  ("left_edge" if left else "right_edge"))})
            else:
                kept.add(line_no)
                kept_by_chr[chrom] += 1
        total = sum(total_by_chr.values())
        if not kept:
            fail("No domains survive the full displacement range in {}.".format(source))
        for chrom in total_by_chr:
            counts.append({"map": label, "chrom": chrom, "original_domains": total_by_chr[chrom],
                           "retained_domains": kept_by_chr[chrom],
                           "excluded_domains": total_by_chr[chrom] - kept_by_chr[chrom],
                           "source_out_of_bounds_domains": source_bad_by_chr[chrom],
                           "shift_edge_excluded_domains": total_by_chr[chrom] - kept_by_chr[chrom] - source_bad_by_chr[chrom]})
        prepared.append((label, source, records, kept, total, sum(source_bad_by_chr.values())))

    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".{}-".format(output.name), dir=str(output.parent)))
    manifest, master_paths = [], []
    try:
        (stage / "beds").mkdir()
        # Numeric offset order; within each offset, preserve the source-list order.
        for offset in offsets:
            for label, source, records, kept, total, source_bad_count in prepared:
                shifted_label = "{}__{}".format(label, offset_tag(offset))
                filename = shifted_label + ".bed"
                final_path = output / "beds" / filename
                with (stage / "beds" / filename).open("x", encoding="utf-8", newline="\n") as handle:
                    for line_no, header, fields in records:
                        if fields is None:
                            handle.write(header + "\n")
                        elif line_no in kept:
                            shifted = list(fields)
                            shifted[1], shifted[2] = str(int(fields[1]) + offset), str(int(fields[2]) + offset)
                            if args.bed_format == "bed12":
                                shifted[6], shifted[7] = str(int(fields[6]) + offset), str(int(fields[7]) + offset)
                            handle.write("\t".join(shifted) + "\n")
                master_paths.append(str(final_path))
                manifest.append({"shifted_label": shifted_label, "map": label, "offset_bp": offset,
                                 "offset_kb": offset // 1000, "bed_path": str(final_path),
                                 "source_bed": str(source), "original_domains": total,
                                 "retained_domains": len(kept), "excluded_domains": total - len(kept),
                                 "source_out_of_bounds_domains": source_bad_count,
                                 "shift_edge_excluded_domains": total - len(kept) - source_bad_count})
        (stage / "TADs.list").write_text("\n".join(master_paths) + "\n", encoding="utf-8")
        write_table(stage / "manifest.tsv", MANIFEST_FIELDS, manifest)
        write_table(stage / "excluded_domains.tsv",
                    ["map", "source_bed", "source_line", "chrom", "start", "end", "chrom_length",
                     "source_left_overhang_bp", "source_right_overhang_bp", "reason"], excluded)
        write_table(stage / "domain_counts.tsv",
                    ["map", "chrom", "original_domains", "retained_domains", "excluded_domains",
                     "source_out_of_bounds_domains", "shift_edge_excluded_domains"], counts)
        config = {
            "tool_version": VERSION, "created_utc": datetime.now(timezone.utc).isoformat(),
            "tads_list": str(tad_list), "chrom_sizes": str(sizes_path), "output_directory": str(output),
            "offsets_bp": offsets, "bed_format": args.bed_format,
            "out_of_bounds_policy": args.out_of_bounds,
            "edge_policy": "same original intervals valid at every offset, including zero",
            "zero_baseline": "zero-offset, common geometric subset",
            "expression_based_eligibility": "not evaluated",
            "source_sha256": {str(path): hash_file(path) for path in
                              [tad_list, sizes_path] + [source for _, source in maps]},
        }
        (stage / "generation.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        if output.exists():
            fail("Output appeared during preparation; nothing was overwritten: {}".format(output))
        stage.rename(output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    print("Prepared {} maps x {} offsets = {} BED files.".format(len(maps), len(offsets), len(manifest)))
    print("Master list: {}".format(output / "TADs.list"))
    print("Manifest:    {}".format(output / "manifest.tsv"))
    for label, source, records, kept, total, source_bad_count in prepared:
        print("{}: retained {}/{} domains; excluded {} ({} source out-of-bounds, "
              "{} additional shift-edge exclusions).".format(
                  label, len(kept), total, total - len(kept), source_bad_count,
                  total - len(kept) - source_bad_count))
    print("Original inputs were not modified. Point do_lists.sh at the master list.")


def read_manifest(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"shifted_label", "map", "offset_bp", "offset_kb"}
        if not required.issubset(reader.fieldnames or []):
            fail("Manifest lacks required fields: {}".format(", ".join(sorted(required))))
        entries, labels, pairs = [], set(), set()
        for row in reader:
            try:
                offset = int(row["offset_bp"])
                offset_kb = int(row["offset_kb"])
            except (TypeError, ValueError):
                fail("Manifest contains a noninteger offset.")
            label, name = row["shifted_label"], row["map"]
            if not label or not name or offset != offset_kb * 1000:
                fail("Manifest contains an empty label or inconsistent offset.")
            if label in labels or (name, offset) in pairs:
                fail("Manifest contains a duplicate label or map-offset pair.")
            labels.add(label)
            pairs.add((name, offset))
            entries.append(row)
    if not entries:
        fail("Manifest has no map-offset entries.")
    return entries


def expected_datasets(path):
    names, seen = [], set()
    for line_no, raw in read_lines(path):
        name = raw.strip()
        if not name or name.startswith("#"):
            continue
        if len(name.split()) != 1 or name in seen:
            fail("{}:{}: expected a unique, whitespace-free result dataset label.".format(path, line_no))
        names.append(name)
        seen.add(name)
    if not names:
        fail("Expected-dataset list is empty.")
    return names


def collect(args):
    results, manifest_path, output = map(resolved, (args.results, args.manifest, args.output))
    if output.exists():
        fail("Output already exists; choose a new --output: {}".format(output))
    manifest = read_manifest(manifest_path)
    lookup = {row["shifted_label"]: row for row in manifest}
    observed, dataset_order = {}, []
    number_names = ["z_e", "p_z_e", "z_c", "p_z_c_left", "dispersion", "concordance"]
    flagged = 0
    non_significant = 0
    for line_no, raw in read_lines(results):
        text = raw.strip()
        if not text or text.startswith("#"):
            continue
        fields = text.split()
        if len(fields) != 8:
            fail("{}:{}: expected 8 fields, found {}.".format(results, line_no, len(fields)))
        dataset, label = fields[:2]
        if label not in lookup:
            fail("{}:{}: map label {!r} is absent from the manifest. Expected the generated "
                 "BED basename without .bed; old/unshifted rows must be kept separately.".format(
                     results, line_no, label))
        key = (dataset, label)
        if key in observed:
            fail("{}:{}: duplicate result for dataset {} and map {}.".format(results, line_no, dataset, label))
        statuses = []
        for index, token in enumerate(fields[2:]):
            if number_names[index] == "concordance" and token.casefold() == "ns":
                # ConGEE reports a category here; preserve it, never infer a value.
                statuses.append("non_significant:concordance")
                non_significant += 1
                continue
            try:
                value = float("nan") if token.lower() in MISSING else float(token)
            except ValueError:
                fail("{}:{}: invalid {} value {!r}.".format(results, line_no, number_names[index], token))
            if not math.isfinite(value):
                statuses.append("unavailable:" + number_names[index])
            elif index in (1, 3) and not 0 <= value <= 1:
                fail("{}:{}: {} is outside [0, 1].".format(results, line_no, number_names[index]))
        if dataset not in dataset_order:
            dataset_order.append(dataset)
        status = ";".join(statuses) or "ok"
        if any(status.startswith("unavailable:") for status in statuses):
            flagged += 1
        observed[key] = (fields[2:], status)

    if args.expected_datasets:
        datasets = expected_datasets(resolved(args.expected_datasets))
        unexpected = sorted(set(dataset_order) - set(datasets))
        if unexpected:
            fail("Unexpected dataset labels: {}".format(", ".join(unexpected)))
    else:
        datasets = dataset_order
    if not datasets:
        fail("No result datasets found. An expected-dataset list is required to audit an empty results file.")
    missing = [(dataset, entry["shifted_label"]) for dataset in datasets for entry in manifest
               if (dataset, entry["shifted_label"]) not in observed]
    if missing and not args.allow_missing:
        examples = "; ".join("{} / {}".format(*pair) for pair in missing[:5])
        fail("{} expected result rows are missing (examples: {}). No output written. "
             "Use --allow-missing to emit explicitly marked missing rows.".format(len(missing), examples))
    rows = []
    for dataset in datasets:
        for entry in manifest:
            label = entry["shifted_label"]
            values, status = observed.get((dataset, label), (["NA"] * 6, "missing_row"))
            row = {"dataset": dataset, "map": entry["map"], "offset_bp": entry["offset_bp"],
                   "offset_kb": entry["offset_kb"], "status": status, "shifted_map": label}
            row.update(zip(number_names, values))
            rows.append(row)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_table(output, RESULT_FIELDS, rows)
    print("Wrote {} rows to {}.".format(len(rows), output))
    print("Observed rows: {}; missing rows: {}; rows with unavailable numbers: {}; "
          "rows with non-significant concordance: {}.".format(
              len(observed), len(missing), flagged, non_significant))
    if not args.expected_datasets:
        print("Completeness was checked within observed datasets only. Entirely absent datasets "
              "require --expected-datasets with exact result labels.")
    print("Input result values were preserved; no p-values, corrections, or direct tests were calculated.")


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="version", version=VERSION)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="Generate shifted BEDs and one master TADs.list.")
    prep.add_argument("--tads-list", default="TADs.list", help="One BED path per line; relative paths use this list's directory.")
    prep.add_argument("--chrom-sizes", default="~/bin/hg38.chr.sizes", help="UCSC chromosome sizes matching the BED assembly.")
    prep.add_argument("--out-dir", default="TAD_offsets", help="New output directory; existing directories are refused.")
    prep.add_argument("--start-kb", type=int, default=-500)
    prep.add_argument("--stop-kb", type=int, default=500)
    prep.add_argument("--step-kb", type=int, default=50, help="Positive grid step. Offset zero is always included.")
    prep.add_argument("--out-of-bounds", choices=["error", "exclude"], default="error",
                      help="error: reject source intervals outside chromosome bounds; exclude: omit them "
                           "from every offset and record their original coordinates/overhangs. Never clip.")
    prep.add_argument("--bed-format", choices=["auto", "interval", "bed12"], default="auto",
                      help="auto: accept 3-6 fields; interval: shift first 3 fields, preserve other annotations; "
                           "bed12: validate BED12 and also shift thickStart/thickEnd.")
    prep.set_defaults(func=prepare)
    coll = commands.add_parser("collect", help="Join the 8-column results to the offset manifest.")
    coll.add_argument("--results", default="results.txt")
    coll.add_argument("--manifest", default="TAD_offsets/manifest.tsv")
    coll.add_argument("--output", default="results.offsets.tsv", help="New output file; existing files are refused.")
    coll.add_argument("--expected-datasets", help="Optional file: one exact result dataset label per line; not inferred from cells.list.")
    coll.add_argument("--allow-missing", action="store_true", help="Include missing combinations as NA with status=missing_row.")
    coll.set_defaults(func=collect)
    return parser


def main():
    args = make_parser().parse_args()
    try:
        args.func(args)
    except (InputError, OSError, UnicodeError, csv.Error) as error:
        print("ERROR: {}".format(error), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
