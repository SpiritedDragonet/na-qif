#!/usr/bin/env python3
import glob
import os
import re
import sys
from pathlib import Path


USAGE = """Usage:
  python convert_mathjax_katex_to_mdmath.py -a [-o [OUT]]
  python convert_mathjax_katex_to_mdmath.py [-o [OUT]] <file_or_glob> [<file_or_glob> ...]

Options:
  -a, --all           Convert all .md files in this script directory.
  -o, --output [OUT]  Save-as. If OUT is given, write to OUT (single input only).
                      If OUT is omitted, append _converted to each input filename.

Notes:
  - Globs like "docs/*report*.md" are supported.
  - Directories are not expanded; use a glob (dir/*.md) instead.
  - Lines that are only "[" or "$" are treated as display-math delimiters.
  - Parentheses with LaTeX-like markers (backslash command, _ or ^) are treated as inline math.
"""

FENCE_START_RE = re.compile(r"^\s*(`{3,}|~{3,})")
INLINE_CODE_RE = re.compile(r"(`+)([^`\n]*?)\1")
INLINE_CODE_TOKEN_RE = re.compile(r"@@INLINE_CODE_(\d+)@@")
MATH_TOKEN_RE = re.compile(r"@@MATH_(\d+)@@")
MATH_HINT_RE = re.compile(r"(\\[A-Za-z]+|[_^])")
URL_HINT_RE = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*://")
DISPLAY_RE = re.compile(r"(?<!\\)\\\[(.*?)\\\]", re.DOTALL)
INLINE_RE = re.compile(r"(?<!\\)\\\(([^\\n]*?)\\\)")
FULLWIDTH_OPEN = "\uFF08"
FULLWIDTH_CLOSE = "\uFF09"


def parse_args(argv):
    all_files = False
    output = None
    paths = []

    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg in ("-h", "--help"):
            print(USAGE)
            sys.exit(0)
        if arg == "--":
            paths.extend(argv[i + 1 :])
            break
        if arg in ("-a", "--all"):
            all_files = True
            i += 1
            continue
        if arg in ("-o", "--output", "--save-as"):
            if i + 1 < len(argv) and not argv[i + 1].startswith("-"):
                output = argv[i + 1]
                i += 2
            else:
                output = ""
                i += 1
            continue
        paths.append(arg)
        i += 1

    return all_files, output, paths


def is_fence_end(line, fence):
    fence_char = fence[0]
    fence_len = len(fence)
    return re.match(rf"^\s*{re.escape(fence_char)}{{{fence_len},}}\s*$", line) is not None


def split_fenced_blocks(text):
    lines = text.splitlines(keepends=True)
    segments = []
    buf = []
    code_buf = []
    in_fence = False
    fence = ""

    for line in lines:
        if not in_fence:
            m = FENCE_START_RE.match(line)
            if m:
                if buf:
                    segments.append(("text", "".join(buf)))
                    buf = []
                in_fence = True
                fence = m.group(1)
                code_buf = [line]
            else:
                buf.append(line)
        else:
            code_buf.append(line)
            if is_fence_end(line, fence):
                segments.append(("code", "".join(code_buf)))
                code_buf = []
                in_fence = False

    if in_fence:
        segments.append(("code", "".join(code_buf)))
    if buf:
        segments.append(("text", "".join(buf)))

    return segments


def mask_inline_code(text):
    code_spans = []

    def repl(match):
        idx = len(code_spans)
        code_spans.append(match.group(0))
        return f"@@INLINE_CODE_{idx}@@"

    masked = INLINE_CODE_RE.sub(repl, text)
    return masked, code_spans


def unmask_inline_code(text, code_spans):
    def repl(match):
        idx = int(match.group(1))
        if idx < len(code_spans):
            return code_spans[idx]
        return match.group(0)

    return INLINE_CODE_TOKEN_RE.sub(repl, text)


def is_escaped(text, idx):
    backslashes = 0
    i = idx - 1
    while i >= 0 and text[i] == "\\":
        backslashes += 1
        i -= 1
    return backslashes % 2 == 1


def find_matching_double_dollar(text, start):
    i = start
    while i < len(text) - 1:
        if text[i] == "$" and text[i + 1] == "$" and not is_escaped(text, i):
            return i
        i += 1
    return -1


def find_matching_single_dollar(text, start):
    i = start
    while i < len(text):
        if text[i] == "$" and not is_escaped(text, i):
            if i + 1 < len(text) and text[i + 1] == "$":
                i += 2
                continue
            return i
        i += 1
    return -1


def mask_math(text):
    spans = []
    out = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == "$" and not is_escaped(text, i):
            if i + 1 < len(text) and text[i + 1] == "$":
                end = find_matching_double_dollar(text, i + 2)
                if end != -1:
                    spans.append(text[i : end + 2])
                    out.append(f"@@MATH_{len(spans) - 1}@@")
                    i = end + 2
                    continue
            else:
                end = find_matching_single_dollar(text, i + 1)
                if end != -1:
                    spans.append(text[i : end + 1])
                    out.append(f"@@MATH_{len(spans) - 1}@@")
                    i = end + 1
                    continue
        out.append(ch)
        i += 1
    return "".join(out), spans


def unmask_math(text, spans):
    def repl(match):
        idx = int(match.group(1))
        if idx < len(spans):
            return spans[idx]
        return match.group(0)

    return MATH_TOKEN_RE.sub(repl, text)


def is_math_like(text):
    if URL_HINT_RE.search(text):
        return False
    return MATH_HINT_RE.search(text) is not None


def convert_inline_parens_in_line(line, open_char, close_char):
    out = []
    stack = []
    last = 0
    start = None

    for i, ch in enumerate(line):
        if ch == open_char:
            if not stack:
                start = i
            stack.append(i)
        elif ch == close_char and stack:
            stack.pop()
            if not stack:
                end = i
                content = line[start + 1 : end]
                is_link = start > 0 and line[start - 1] == "]"
                if not is_link and is_math_like(content):
                    out.append(line[last:start] + "$" + content + "$")
                else:
                    out.append(line[last : end + 1])
                last = end + 1

    out.append(line[last:])
    return "".join(out)


def convert_inline_parens(text):
    lines = text.splitlines(keepends=True)
    out_lines = []
    for line in lines:
        updated = convert_inline_parens_in_line(line, "(", ")")
        updated = convert_inline_parens_in_line(updated, FULLWIDTH_OPEN, FULLWIDTH_CLOSE)
        out_lines.append(updated)
    return "".join(out_lines)


def convert_block_delims(text):
    lines = text.splitlines(keepends=True)
    out_lines = []
    in_block = None
    block_indent = ""

    for line in lines:
        stripped = line.strip()
        line_ending = "\n" if line.endswith("\n") else ""

        if in_block:
            if in_block == "square" and stripped == "]":
                out_lines.append(f"{block_indent}$$" + line_ending)
                in_block = None
                block_indent = ""
            elif in_block == "single_dollar" and stripped == "$":
                out_lines.append(f"{block_indent}$$" + line_ending)
                in_block = None
                block_indent = ""
            else:
                out_lines.append(line)
            continue

        if stripped == "[":
            block_indent = line[: len(line) - len(line.lstrip())]
            out_lines.append(f"{block_indent}$$" + line_ending)
            in_block = "square"
            continue

        if stripped == "$":
            block_indent = line[: len(line) - len(line.lstrip())]
            out_lines.append(f"{block_indent}$$" + line_ending)
            in_block = "single_dollar"
            continue

        out_lines.append(line)

    return "".join(out_lines)


def convert_text(text):
    segments = split_fenced_blocks(text)
    out_parts = []

    for kind, segment in segments:
        if kind == "code":
            out_parts.append(segment)
            continue

        segment = convert_block_delims(segment)
        masked, code_spans = mask_inline_code(segment)
        converted = DISPLAY_RE.sub(lambda m: "$$" + m.group(1) + "$$", masked)
        converted = INLINE_RE.sub(lambda m: "$" + m.group(1) + "$", converted)
        converted, math_spans = mask_math(converted)
        converted = convert_inline_parens(converted)
        converted = unmask_math(converted, math_spans)
        converted = unmask_inline_code(converted, code_spans)
        out_parts.append(converted)

    return "".join(out_parts)


def expand_input(arg):
    arg = os.path.expanduser(arg)
    if any(ch in arg for ch in "*?[]"):
        return [Path(p) for p in glob.glob(arg, recursive=True)]
    p = Path(arg)
    if p.exists():
        return [p]
    return []


def collect_targets(all_files, paths, script_dir):
    candidates = []
    missing = []

    if all_files:
        candidates.extend(script_dir.glob("*.md"))

    for item in paths:
        expanded = expand_input(item)
        if not expanded:
            missing.append(item)
        else:
            candidates.extend(expanded)

    targets = []
    seen = set()
    for path in candidates:
        if path.is_dir():
            print(f"Skip directory (use a glob like {path}{os.sep}*.md): {path}", file=sys.stderr)
            continue
        if path.suffix.lower() != ".md":
            print(f"Skip non-md file: {path}", file=sys.stderr)
            continue
        resolved = str(path.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        targets.append(path)

    return targets, missing


def output_path_for(input_path, output_opt, multiple_inputs):
    if output_opt is None:
        return input_path
    if output_opt == "":
        return input_path.with_name(input_path.stem + "_converted" + input_path.suffix)

    out = Path(os.path.expanduser(output_opt))
    if multiple_inputs:
        raise ValueError("Output path can only be used with a single input file.")

    if out.exists() and out.is_dir():
        return out / (input_path.stem + "_converted" + input_path.suffix)
    return out


def main():
    all_files, output_opt, paths = parse_args(sys.argv[1:])

    script_dir = Path(__file__).resolve().parent
    targets, missing = collect_targets(all_files, paths, script_dir)

    if missing:
        for item in missing:
            print(f"No matches for: {item}", file=sys.stderr)

    if not targets:
        print(USAGE)
        sys.exit(1)

    if output_opt not in (None, "") and len(targets) > 1:
        print("Error: -o/--output with a path only works with a single input file.", file=sys.stderr)
        sys.exit(2)

    converted_count = 0
    for input_path in targets:
        try:
            original = input_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            print(f"Skip non-utf8 file: {input_path}", file=sys.stderr)
            continue

        converted = convert_text(original)
        out_path = output_path_for(input_path, output_opt, len(targets) > 1)

        if converted == original and out_path == input_path:
            print(f"No changes: {input_path}")
            continue

        if not out_path.parent.exists():
            print(f"Output directory does not exist: {out_path.parent}", file=sys.stderr)
            continue

        out_path.write_text(converted, encoding="utf-8", newline="")
        converted_count += 1
        if out_path == input_path:
            print(f"Converted in place: {input_path}")
        else:
            print(f"Converted: {input_path} -> {out_path}")

    if converted_count == 0:
        print("No files converted.")


if __name__ == "__main__":
    main()
