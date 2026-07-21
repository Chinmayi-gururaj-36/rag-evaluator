"""
Chunking script for the RAG evaluator project.

Splits CS229 supervised learning (ch 1-4) and deep learning (ch 7) notes
into section-based chunks, tagged with metadata for traceability.

Input: raw text extracted from the two scoped PDFs (via pdftotext -layout)
Output: chunks.json — a list of chunk objects the RAG pipeline will
        embed and retrieve from.
"""

import re
import json

MAX_CHUNK_CHARS = 2200   # roughly ~400-500 tokens; keeps formulas intact
MIN_CHUNK_CHARS = 200    # merge tiny leftover sections into neighbors

SOURCES = [
    {
        "file": "supervised_learning_ch1-4.txt",
        "source_pdf": "supervised_learning_ch1-4.pdf",
        "chapter_pattern": re.compile(r"^Chapter (\d+)$"),
        "section_pattern": re.compile(r"^([1-4])\.(\d+)\s+([A-Za-z].+)$"),
    },
    {
        "file": "deep_learning_ch7.txt",
        "source_pdf": "deep_learning_ch7.pdf",
        "chapter_pattern": re.compile(r"^Chapter (\d+)$"),
        "section_pattern": re.compile(r"^(7)\.(\d+)\s+([A-Za-z].+)$"),
    },
]


def clean_line(line: str) -> str:
    # de-hyphenate line-wrap artifacts like "algo-\nrithm" -> collapse later
    return line.rstrip("\n")


def load_lines(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [clean_line(l) for l in f.readlines()]


def find_chapter_title(lines, chapter_line_idx):
    # Title is usually the very next non-empty line after "Chapter N"
    for j in range(chapter_line_idx + 1, min(chapter_line_idx + 4, len(lines))):
        if lines[j].strip():
            return lines[j].strip()
    return "Untitled"


def split_into_sections(lines, section_pattern, chapter_pattern):
    """Returns list of dicts: {chapter_num, chapter_title, section_num,
    section_title, start_line, end_line}"""
    chapter_num = None
    chapter_title = None
    sections = []
    current = None

    for i, line in enumerate(lines):
        cm = chapter_pattern.match(line.strip())
        if cm:
            chapter_num = cm.group(1)
            chapter_title = find_chapter_title(lines, i)
            continue

        sm = section_pattern.match(line.strip())
        if sm:
            if current is not None:
                current["end_line"] = i
                sections.append(current)
            current = {
                "chapter_num": chapter_num,
                "chapter_title": chapter_title,
                "section_num": f"{sm.group(1)}.{sm.group(2)}",
                "section_title": sm.group(3).strip().rstrip("-"),
                "start_line": i,
                "end_line": None,
            }

    if current is not None:
        current["end_line"] = len(lines)
        sections.append(current)

    return sections


def sections_to_text_chunks(lines, sections, source_pdf):
    chunks = []
    for sec in sections:
        body_lines = lines[sec["start_line"] : sec["end_line"]]
        text = "\n".join(body_lines).strip()
        text = re.sub(r"-\n", "", text)   # de-hyphenate wrapped words
        text = re.sub(r"\n{2,}", "\n\n", text)
        if len(text) < MIN_CHUNK_CHARS:
            continue

        # Split long sections into sub-chunks by greedily packing lines,
        # preferring to break after a line that ends a sentence (ends in
        # '.', ':', or a closed equation) rather than mid-derivation.
        if len(text) <= MAX_CHUNK_CHARS:
            sub_texts = [text]
        else:
            body_only_lines = text.split("\n")
            sub_texts, buf_lines, buf_len = [], [], 0
            for ln in body_only_lines:
                projected = buf_len + len(ln) + 1
                good_break_point = buf_lines and buf_lines[-1].rstrip().endswith((".", ":"))
                if projected > MAX_CHUNK_CHARS and buf_lines and (good_break_point or projected > MAX_CHUNK_CHARS * 1.3):
                    sub_texts.append("\n".join(buf_lines).strip())
                    buf_lines, buf_len = [], 0
                buf_lines.append(ln)
                buf_len += len(ln) + 1
            if buf_lines:
                sub_texts.append("\n".join(buf_lines).strip())
            # merge any tiny trailing fragment into the previous chunk
            if len(sub_texts) > 1 and len(sub_texts[-1]) < MIN_CHUNK_CHARS:
                sub_texts[-2] = sub_texts[-2] + "\n" + sub_texts[-1]
                sub_texts.pop()

        for idx, sub_text in enumerate(sub_texts):
            part_suffix = f"_part{idx+1}" if len(sub_texts) > 1 else ""
            chunks.append({
                "source_pdf": source_pdf,
                "chapter_num": sec["chapter_num"],
                "chapter_title": sec["chapter_title"],
                "section_num": sec["section_num"],
                "section_title": sec["section_title"],
                "part": idx + 1,
                "total_parts": len(sub_texts),
                "text": sub_text,
                "char_count": len(sub_text),
            })
    return chunks


def main():
    all_chunks = []
    for src in SOURCES:
        lines = load_lines(src["file"])
        sections = split_into_sections(lines, src["section_pattern"], src["chapter_pattern"])
        chunks = sections_to_text_chunks(lines, sections, src["source_pdf"])
        all_chunks.extend(chunks)

    # Assign final sequential IDs
    for i, c in enumerate(all_chunks):
        c["chunk_id"] = f"chunk_{i+1:03d}"

    # Reorder keys nicely
    ordered = []
    for c in all_chunks:
        ordered.append({
            "chunk_id": c["chunk_id"],
            "source_pdf": c["source_pdf"],
            "chapter": f"{c['chapter_num']} - {c['chapter_title']}",
            "section": f"{c['section_num']} {c['section_title']}",
            "part": f"{c['part']}/{c['total_parts']}",
            "char_count": c["char_count"],
            "text": c["text"],
        })

    with open("chunks.json", "w", encoding="utf-8") as f:
        json.dump(ordered, f, indent=2)

    print(f"Total chunks: {len(ordered)}")
    print(f"Avg chunk size: {sum(c['char_count'] for c in ordered) / len(ordered):.0f} chars")
    print("\nChunks per section:")
    for c in ordered:
        print(f"  {c['chunk_id']}  [{c['section']}]  part {c['part']}  ({c['char_count']} chars)")


if __name__ == "__main__":
    main()
