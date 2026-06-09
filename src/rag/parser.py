from __future__ import annotations


def parse_policy_markdown(markdown_text: str) -> list[dict]:
    """Chunk the policy markdown by H2 + H3 + H3 content.

    Sections without an H3 are kept as one H2-level chunk so general policy
    content is still searchable, but all H3 sections follow the required lab
    structure exactly.
    """
    chunks: list[dict] = []
    current_h2: str | None = None
    h2_intro: list[str] = []
    current_h3: str | None = None
    current_content: list[str] = []

    def flush_h3() -> None:
        nonlocal current_content, current_h3
        if not current_h2 or not current_h3:
            return
        content = "\n".join(line.rstrip() for line in current_content).strip()
        if not content:
            return
        citation = f"policy_mock_vi.md > {current_h2} > {current_h3}"
        chunks.append(
            {
                "section_h2": current_h2,
                "section_h3": current_h3,
                "citation": citation,
                "content": content,
                "rendered_text": f"## {current_h2}\n### {current_h3}\n{content}",
            }
        )

    def flush_h2_intro() -> None:
        nonlocal h2_intro
        if not current_h2 or not h2_intro:
            return
        content = "\n".join(line.rstrip() for line in h2_intro).strip()
        if not content:
            h2_intro = []
            return
        citation = f"policy_mock_vi.md > {current_h2}"
        chunks.append(
            {
                "section_h2": current_h2,
                "section_h3": "",
                "citation": citation,
                "content": content,
                "rendered_text": f"## {current_h2}\n{content}",
            }
        )
        h2_intro = []

    for raw_line in markdown_text.splitlines():
        line = raw_line.rstrip()
        if line.startswith("## ") and not line.startswith("### "):
            flush_h3()
            flush_h2_intro()
            current_h2 = line[3:].strip()
            h2_intro = []
            current_h3 = None
            current_content = []
            continue
        if line.startswith("### "):
            flush_h3()
            flush_h2_intro()
            current_h3 = line[4:].strip()
            current_content = []
            continue
        if current_h3:
            current_content.append(line)
        elif current_h2:
            h2_intro.append(line)

    flush_h3()
    flush_h2_intro()
    return chunks
