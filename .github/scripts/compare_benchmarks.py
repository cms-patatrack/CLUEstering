#!/usr/bin/env python3
"""Compare Google Benchmark JSON outputs (PR build vs. release baseline) and
render a markdown report. Stdlib-only, used by the benchmark-*-gpu.yml
workflows' "compare" job.
"""

import argparse
import json
import sys

DEFAULT_THRESHOLD_PCT = 10.0


def load_benchmarks(path):
    with open(path) as f:
        data = json.load(f)
    out = {}
    for entry in data.get("benchmarks", []):
        name = entry.get("name")
        if name is None:
            continue
        out[name] = {
            "real_time": entry.get("real_time"),
            "time_unit": entry.get("time_unit", "?"),
        }
    return out


def format_delta(baseline_time, pr_time):
    if baseline_time in (None, 0):
        return None
    return (pr_time - baseline_time) / baseline_time * 100.0


def render_backend_section(label, pr_path, baseline_path, baseline_label, threshold):
    pr = load_benchmarks(pr_path)
    baseline = load_benchmarks(baseline_path)

    common = sorted(set(pr) & set(baseline))
    pr_only = sorted(set(pr) - set(baseline))
    baseline_only = sorted(set(baseline) - set(pr))

    lines = [f"### {label}", ""]

    if not common:
        lines.append("_No benchmarks in common between PR and baseline._")
    else:
        lines.append(f"| Benchmark | {baseline_label} | PR | Delta | |")
        lines.append("|---|---:|---:|---:|:---:|")
        warnings = 0
        for name in common:
            b = baseline[name]
            p = pr[name]
            unit = p["time_unit"] or b["time_unit"]
            if p["time_unit"] != b["time_unit"]:
                lines.append(
                    f"| {name} | different units ({b['time_unit']} vs {p['time_unit']}) | | | ⚠️ |"
                )
                warnings += 1
                continue
            delta = format_delta(b["real_time"], p["real_time"])
            if delta is None:
                lines.append(
                    f"| {name} | {b['real_time']:.2f} {unit} | {p['real_time']:.2f} {unit} | N/A | |"
                )
                continue
            flag = "⚠️" if abs(delta) > threshold else "✅"
            if flag == "⚠️":
                warnings += 1
            sign = "+" if delta >= 0 else ""
            lines.append(
                f"| {name} | {b['real_time']:.2f} {unit} | {p['real_time']:.2f} {unit} "
                f"| {sign}{delta:.2f}% | {flag} |"
            )
        lines.append("")
        if warnings:
            lines.append(
                f"**{warnings} benchmark(s) beyond the {threshold:.0f}% threshold relative to {baseline_label}.**"
            )
        else:
            lines.append(f"No variation beyond the {threshold:.0f}% threshold.")

    if pr_only:
        lines.append("")
        lines.append(f"Benchmarks only in the PR (missing in {baseline_label}): " + ", ".join(pr_only))
    if baseline_only:
        lines.append("")
        lines.append(f"Benchmarks only in {baseline_label} (removed in the PR): " + ", ".join(baseline_only))

    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair",
        action="append",
        nargs=3,
        metavar=("LABEL", "PR_JSON", "BASELINE_JSON"),
        required=True,
        help="One backend to compare: name, PR JSON file, baseline JSON file. "
        "Repeatable to compare multiple backends (e.g. serial, openmp, cuda).",
    )
    parser.add_argument(
        "--baseline-label",
        default="baseline",
        help="Label for the baseline shown in the table (e.g. the release tag, v2.11.0).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD_PCT,
        help=f"Percentage delta threshold above which a benchmark is flagged with ⚠️ (default {DEFAULT_THRESHOLD_PCT}).",
    )
    parser.add_argument(
        "--output",
        help="If given, also write the report to this file in addition to stdout.",
    )
    args = parser.parse_args()

    sections = [
        render_backend_section(label, pr_json, baseline_json, args.baseline_label, args.threshold)
        for label, pr_json, baseline_json in args.pair
    ]
    report = f"## GPU benchmark comparison vs {args.baseline_label}\n\n" + "\n".join(sections)

    print(report)
    if args.output:
        with open(args.output, "w") as f:
            f.write(report + "\n")


if __name__ == "__main__":
    sys.exit(main())
