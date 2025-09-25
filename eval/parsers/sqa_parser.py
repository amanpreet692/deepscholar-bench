import os
import json
from collections import OrderedDict
import re

try:
    from parsers.parser import Parser, ParserType
    from parse_generated_text import (
        replace_refs,
        replace_latex_cites,
        get_arxiv_title_and_abstract,
    )
except ImportError:
    from ..parsers.parser import Parser, ParserType
    from ..parse_generated_text import (
        replace_refs,
        replace_latex_cites,
        get_arxiv_title_and_abstract,
    )


class SQAParser(Parser):
    parser_type = ParserType.SCHOLARQA

    @property
    def citation_pattern(self):
        return re.compile(r"\[(\d+)\]\((\d+)\)")

    def _load_file(self):
        import os
        json_files = [
            f for f in os.listdir(self.folder_path) if f.endswith(".json")
        ]
        if not json_files:
            return
        file_path = os.path.join(self.folder_path, json_files[0])
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        sections = data.get("sections", "[]")
        raw_text, ctxs = "", dict()
        for sec in sections:
            raw_text += sec["title"]+sec["text"]+"\n"
            ctxs.update({c["id"]: c for c in sec["citations"]})
        ctxs = list(ctxs.values())
        reference_map = {
            str(i): {
                "url": f"[{i}]({ctx['paper']['corpus_id']})",
                "title": ctx["paper"].get("title", ""),
                "text": "...".join(ctx["snippets"]),
            }
            for i, ctx in enumerate(ctxs)
        }
        for i, ctx in enumerate(ctxs):
            raw_text = raw_text.replace(ctx['id'], f"[{i}]")
        updated_text = replace_refs(raw_text, reference_map)
        self.clean_text, self.docs = self._to_autoais(updated_text, reference_map)
        if len(self.docs) == 0:
            updated_text = replace_latex_cites(updated_text, reference_map)
            self.clean_text, self.docs = self._to_autoais(updated_text)
        self.raw_generated_text = updated_text

        self.citations_for_cite_quality = [(d["title"], d["sent"]) for d in self.docs]

    def _to_autoais(self, updated_text: str, reference_map: dict):
        url2abs = reference_map or {}
        link2id: OrderedDict[str, int] = OrderedDict()
        docs = []

        def repl(match: re.Match) -> str:
            index, _ = match.groups()
            if index not in link2id:
                link2id[index] = len(link2id) + 1
                info = url2abs.get(index, {})
                docs.append(
                    {"title": info.get("title", ""), "sent": info.get("text", "")}
                )
            return f"[{link2id[index]}]"

        clean_text = self.citation_pattern.sub(repl, updated_text).strip()
        return clean_text, docs

    def _process_for_cite_quality(self) -> tuple[str, list[dict]]:
        md_files = [f for f in os.listdir(self.folder_path) if f.endswith(".md")]
        md_file_path = os.path.join(self.folder_path, md_files[0])
        with open(md_file_path, "r", encoding="utf-8") as f:
            content = f.read()

        if "References" in content:
            related_work_section = content.split("References")[0]
        else:
            related_work_section = content

        reference_section = ""
        if "References" in content:
            reference_section = content.split("References")[1]

        arxiv_url_pattern = r"\[(\d+)\]\s*(https://arxiv\.org/abs/(\d+\.\d+(?:v\d+)?))"
        arxiv_url_matches = re.findall(arxiv_url_pattern, reference_section)

        search_ai_arxiv_pattern = (
            r"\[(\d+)\]\s*(.+?)[.:]\s*http://arxiv\.org/abs/(\d+\.\d+(?:v\d+)?)"
        )
        search_ai_arxiv_matches = re.findall(search_ai_arxiv_pattern, reference_section)

        # Pattern for SearchAI format without punctuation: [1] Title http://arxiv.org/abs/ID
        search_ai_arxiv_pattern_no_punct = (
            r"\[(\d+)\]\s*(.+?)\s+http://arxiv\.org/abs/(\d+\.\d+(?:v\d+)?)"
        )
        search_ai_arxiv_matches_no_punct = re.findall(
            search_ai_arxiv_pattern_no_punct, reference_section
        )

        # Pattern for arXiv: prefix format: [1] Title. arXiv:2501.05098v1.
        arxiv_prefix_pattern = r"\[(\d+)\]\s*(.+?)[.:]\s*arXiv:(\d+\.\d+(?:v\d+)?)"
        arxiv_prefix_matches = re.findall(arxiv_prefix_pattern, reference_section)

        # Pattern for arXiv: prefix without punctuation: [1] Title arXiv:2501.05098v1
        arxiv_prefix_no_punct_pattern = r"\[(\d+)\]\s*(.+?)\s+arXiv:(\d+\.\d+(?:v\d+)?)"
        arxiv_prefix_no_punct_matches = re.findall(
            arxiv_prefix_no_punct_pattern, reference_section
        )

        # Pattern for numbered list format: 1. Title. http://arxiv.org/abs/ID
        numbered_list_pattern = (
            r"(\d+)\.\s*(.+?)[.:]\s*http://arxiv\.org/abs/(\d+\.\d+(?:v\d+)?)"
        )
        numbered_list_matches = re.findall(numbered_list_pattern, reference_section)

        # Pattern for numbered list without punctuation: 1. Title http://arxiv.org/abs/ID
        numbered_list_no_punct_pattern = (
            r"(\d+)\.\s*(.+?)\s+http://arxiv\.org/abs/(\d+\.\d+(?:v\d+)?)"
        )
        numbered_list_no_punct_matches = re.findall(
            numbered_list_no_punct_pattern, reference_section
        )

        # Pattern for numbered list with just URL: 1. http://arxiv.org/abs/ID
        numbered_list_url_only_pattern = (
            r"(\d+)\.\s*http://arxiv\.org/abs/(\d+\.\d+(?:v\d+)?)"
        )
        numbered_list_url_only_matches = re.findall(
            numbered_list_url_only_pattern, reference_section
        )

        # Pattern for numbered list with arXiv prefix: 1. Title. arXiv:ID
        numbered_list_arxiv_prefix_pattern = (
            r"(\d+)\.\s*(.+?)[.:]\s*arXiv:(\d+\.\d+(?:v\d+)?)"
        )
        numbered_list_arxiv_prefix_matches = re.findall(
            numbered_list_arxiv_prefix_pattern, reference_section
        )

        # Pattern for numbered list with just arXiv prefix: 1. arXiv:ID
        numbered_list_arxiv_prefix_only_pattern = r"(\d+)\.\s*arXiv:(\d+\.\d+(?:v\d+)?)"
        numbered_list_arxiv_prefix_only_matches = re.findall(
            numbered_list_arxiv_prefix_only_pattern, reference_section
        )

        # Combine all matches
        all_arxiv_matches = (
            arxiv_url_matches
            + search_ai_arxiv_matches
            + search_ai_arxiv_matches_no_punct
            + arxiv_prefix_matches
            + arxiv_prefix_no_punct_matches
            + numbered_list_matches
            + numbered_list_no_punct_matches
            + numbered_list_url_only_matches
            + numbered_list_arxiv_prefix_matches
            + numbered_list_arxiv_prefix_only_matches
        )

        # Create reference map and docs
        reference_map = {}
        docs = []

        # Process each arXiv reference
        for match in all_arxiv_matches:
            if (
                len(match) == 3
            ):  # (ref_num, title, arxiv_id) or (ref_num, url, arxiv_id)
                ref_num, title_or_url, arxiv_id = match

                # Check if it's a URL or title
                if title_or_url.startswith("http"):
                    # It's a URL, extract title from arXiv
                    # url = title_or_url
                    try:
                        fetched_title, fetched_abstract = get_arxiv_title_and_abstract(
                            arxiv_id
                        )
                        if fetched_title and fetched_abstract:
                            title = fetched_title
                            abstract = fetched_abstract
                        else:
                            title = f"arXiv:{arxiv_id}"
                            abstract = ""
                    except Exception:
                        title = f"arXiv:{arxiv_id}"
                        abstract = ""
                else:
                    # It's a title, get abstract from arXiv
                    title = title_or_url.strip()
                    try:
                        fetched_title, fetched_abstract = get_arxiv_title_and_abstract(
                            arxiv_id
                        )
                        abstract = fetched_abstract or ""
                    except Exception:
                        abstract = ""

            elif len(match) == 2:  # (ref_num, arxiv_id) - URL only or arXiv prefix only
                ref_num, arxiv_id = match
                try:
                    fetched_title, fetched_abstract = get_arxiv_title_and_abstract(
                        arxiv_id
                    )
                    if fetched_title and fetched_abstract:
                        title = fetched_title
                        abstract = fetched_abstract
                    else:
                        title = f"arXiv:{arxiv_id}"
                        abstract = ""
                except Exception:
                    title = f"arXiv:{arxiv_id}"
                    abstract = ""

            # Add to reference map
            reference_map[ref_num] = {
                "url": f"https://arxiv.org/abs/{arxiv_id}",
                "title": title,
                "text": abstract,
            }

            # Add to docs for evaluation
            docs.append({"title": title, "sent": abstract})

        # Process the related work section to replace citations
        def process_openscholar_citations(text, reference_map):
            """Process openscholar citations in format [1], [2], etc."""
            import re
            from collections import OrderedDict

            link2id = OrderedDict()

            def repl(match):
                ref_num = match.group(1)
                if ref_num in reference_map:
                    if ref_num not in link2id:
                        link2id[ref_num] = len(link2id) + 1
                    return f"[{link2id[ref_num]}]"
                else:
                    return match.group(0)

            pattern = r"\[(\d+)\]"
            clean_text = re.sub(pattern, repl, text)

            return clean_text

        updated_text = process_openscholar_citations(
            related_work_section, reference_map
        )
        answer = updated_text

        # Fallback: if docs is empty, create docs from reference_map
        if not docs or len(docs) == 0:
            docs = []
            for ref_id, ref_info in reference_map.items():
                docs.append({"title": ref_info["title"], "text": ref_info["text"]})

            # Update the text to use sequential citations [1], [2], etc.
            citation_mapping = {}  # Map old citations to new indices
            for i, (ref_id, ref_info) in enumerate(reference_map.items()):
                citation_mapping[ref_id] = str(i + 1)

            answer = updated_text
            for old_ref, new_ref in citation_mapping.items():
                answer = answer.replace(f"[{old_ref}]", f"[{new_ref}]")

        return answer, [(doc["title"], doc["sent"]) for doc in docs]  # type: ignore
