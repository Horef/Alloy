from __future__ import annotations

import html
import json
from collections import Counter, defaultdict
from pathlib import Path

from .models import EvaluationRecord, Outcome


def build_summary(records: list[EvaluationRecord]) -> dict:
    counts = Counter(record.outcome.value for record in records)
    by_topic: dict[str, Counter] = defaultdict(Counter)
    for record in records:
        by_topic[record.question.topic][record.outcome.value] += 1
    scored = [record for record in records if record.scores]
    answerable = sum(record.question.answerable for record in records)
    correct = counts[Outcome.CORRECT_ANSWER.value]
    return {
        "total": len(records),
        "answerable_questions": answerable,
        "unanswerable_questions": len(records) - answerable,
        "correct_answer_rate_on_answerable": correct / answerable if answerable else None,
        "outcomes": dict(counts),
        "average_scores": {
            name: (sum(getattr(r.scores, name) for r in scored) / len(scored) if scored else None)
            for name in ("correctness", "completeness", "relevance", "groundedness")
        },
        "by_topic": {topic: dict(values) for topic, values in sorted(by_topic.items())},
    }


def write_report(records: list[EvaluationRecord], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = build_summary(records)
    json_path = output_dir / "evaluation_summary.json"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    outcome_rows = "".join(f"<tr><td>{html.escape(name)}</td><td>{count}</td></tr>" for name, count in summary["outcomes"].items())
    detail_rows = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in (
            r.question.id, r.question.topic, r.question.question, r.outcome.value,
            r.scores.explanation if r.scores else (r.result.error or r.judge_error),
        )) + "</tr>" for r in records
    )
    rate = summary["correct_answer_rate_on_answerable"]
    rate_text = "n/a" if rate is None else f"{rate:.1%}"
    document = f"""<!doctype html><html><head><meta charset="utf-8"><title>Chatbot evaluation</title>
<style>body{{font:15px system-ui;margin:2rem;color:#172033}}.cards{{display:flex;gap:1rem;flex-wrap:wrap}}.card{{background:#eef4ff;padding:1rem 1.4rem;border-radius:10px}}table{{border-collapse:collapse;width:100%;margin:1rem 0}}th,td{{border:1px solid #ccd4e0;padding:.55rem;text-align:left;vertical-align:top}}th{{background:#172033;color:white}}tr:nth-child(even){{background:#f7f9fc}}</style></head>
<body><h1>Chatbot evaluation</h1><div class="cards"><div class="card"><b>Total</b><br>{summary['total']}</div><div class="card"><b>Correct answer rate</b><br>{rate_text}</div><div class="card"><b>Answerable / unanswerable</b><br>{summary['answerable_questions']} / {summary['unanswerable_questions']}</div></div>
<h2>Outcome counts</h2><table><tr><th>Outcome</th><th>Count</th></tr>{outcome_rows}</table>
<h2>Question details</h2><table><tr><th>ID</th><th>Topic</th><th>Question</th><th>Outcome</th><th>Explanation / error</th></tr>{detail_rows}</table></body></html>"""
    html_path = output_dir / "evaluation_report.html"
    html_path.write_text(document, encoding="utf-8")
    return json_path, html_path
