from __future__ import annotations

import html
import json
from collections import Counter, defaultdict
from pathlib import Path

from .labels import OUTCOME_COLORS, OUTCOME_HEBREW, SCORE_HEBREW
from .models import EvaluationRecord, Outcome

ANSWER_SUCCESS = {Outcome.CORRECT_ANSWER}
ANSWER_USEFUL = {Outcome.CORRECT_ANSWER, Outcome.PARTIAL_TOO_LITTLE, Outcome.PARTIAL_TOO_MUCH}
PIPELINE_SUCCESS = {Outcome.CORRECT_ANSWER, Outcome.CORRECT_ABSTENTION}
RISKY = {Outcome.MISLEADING_HALLUCINATION, Outcome.SHOULD_HAVE_ABSTAINED}


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _mean(records: list[EvaluationRecord], field: str, *, exclude_zero: bool = False) -> float | None:
    values = [getattr(record.scores, field) for record in records if record.scores]
    if exclude_zero:
        values = [value for value in values if value > 0]
    return sum(values) / len(values) if values else None


def _retrieval_good(record: EvaluationRecord) -> bool:
    return bool(record.scores and record.scores.retrieval_relevance >= 3 and record.scores.retrieval_correctness >= 3 and record.scores.retrieval_completeness >= 3)


def build_summary(records: list[EvaluationRecord]) -> dict:
    counts = Counter(record.outcome.value for record in records)
    evaluable = [r for r in records if r.outcome not in {Outcome.CHATBOT_ERROR, Outcome.JUDGE_ERROR}]
    answerable = [r for r in evaluable if r.question.answerable]
    retrieval_scored = [r for r in evaluable if r.scores and r.scores.retrieval_relevance > 0]
    good_retrieval = [r for r in retrieval_scored if _retrieval_good(r)]
    generation_failures = [r for r in good_retrieval if r.outcome not in PIPELINE_SUCCESS]
    pipeline = Counter()
    for record in evaluable:
        if not record.scores or record.scores.retrieval_relevance == 0:
            pipeline["retrieval_unavailable"] += 1
        else:
            retrieval = "good" if _retrieval_good(record) else "poor"
            answer = "good" if record.outcome in PIPELINE_SUCCESS else "poor"
            pipeline[f"retrieval_{retrieval}_answer_{answer}"] += 1

    by_topic_records: dict[str, list[EvaluationRecord]] = defaultdict(list)
    for record in records:
        by_topic_records[record.question.topic or "לא סווג"].append(record)
    by_topic = {}
    for topic, topic_records in sorted(by_topic_records.items(), key=lambda item: (-len(item[1]), item[0])):
        topic_evaluable = [r for r in topic_records if r.outcome not in {Outcome.CHATBOT_ERROR, Outcome.JUDGE_ERROR}]
        by_topic[topic] = {
            "total": len(topic_records),
            "correct": sum(r.outcome in ANSWER_SUCCESS for r in topic_records),
            "useful": sum(r.outcome in ANSWER_USEFUL for r in topic_records),
            "risky": sum(r.outcome in RISKY for r in topic_records),
            "good_retrieval": sum(_retrieval_good(r) for r in topic_records),
            "correct_rate": _rate(sum(r.outcome in ANSWER_SUCCESS for r in topic_records), len(topic_evaluable)),
            "risky_rate": _rate(sum(r.outcome in RISKY for r in topic_records), len(topic_evaluable)),
            "outcomes": dict(Counter(r.outcome.value for r in topic_records)),
        }
    return {
        "total": len(records), "evaluable": len(evaluable),
        "answerable_questions": len(answerable),
        "unanswerable_questions": sum(not r.question.answerable for r in evaluable),
        "infrastructure_errors": len(records) - len(evaluable),
        "correct_answer_rate_on_answerable": _rate(sum(r.outcome in ANSWER_SUCCESS for r in answerable), len(answerable)),
        "useful_answer_rate_on_answerable": _rate(sum(r.outcome in ANSWER_USEFUL for r in answerable), len(answerable)),
        "risky_misinformation_rate": _rate(sum(r.outcome in RISKY for r in evaluable), len(evaluable)),
        "retrieval_evaluated": len(retrieval_scored),
        "good_retrieval_rate": _rate(len(good_retrieval), len(retrieval_scored)),
        "generation_failures_despite_good_retrieval": len(generation_failures),
        "generation_failure_question_ids": [r.question.id for r in generation_failures],
        "outcomes": dict(counts),
        "average_scores": {field: _mean(evaluable, field, exclude_zero=field.startswith("retrieval_")) for field in SCORE_HEBREW},
        "pipeline": dict(pipeline), "by_topic": by_topic,
    }


def _pct(value: float | None) -> str:
    return "לא זמין" if value is None else f"{value:.1%}"


def _esc(value: object) -> str:
    return html.escape(str(value or ""))


def _bar(label: str, count: int, total: int, color: str) -> str:
    width = 0 if total == 0 else count / total * 100
    return f'<div class="bar-row"><span>{_esc(label)}</span><div class="track"><i style="width:{width:.2f}%;background:{color}"></i></div><b>{count}</b><small>{width:.1f}%</small></div>'


def _score_bar(label: str, value: float | None, color: str) -> str:
    width = 0 if value is None else value / 4 * 100
    shown = "—" if value is None else f"{value:.2f}/4"
    return f'<div class="score-row"><span>{_esc(label)}</span><div class="track"><i style="width:{width:.2f}%;background:{color}"></i></div><b>{shown}</b></div>'


def _details(record: EvaluationRecord) -> str:
    scores = record.scores
    label, color = OUTCOME_HEBREW[record.outcome], OUTCOME_COLORS[record.outcome]
    score_cells = "" if not scores else "".join(f'<span><b>{_esc(SCORE_HEBREW[field])}</b> {getattr(scores, field)}/4</span>' for field in SCORE_HEBREW)
    if scores:
        explanation = scores.explanation
    elif record.result.error:
        explanation = f"שגיאת מערכת בצ׳אטבוט: {record.result.error}"
    else:
        explanation = f"שגיאה בתהליך הבדיקה: {record.judge_error}"
    retrieval_explanation = scores.retrieval_explanation if scores else ""
    missing = scores.missing_or_wrong if scores else ""
    search = " ".join((record.question.id, record.question.topic, record.question.question, record.result.answer, label)).casefold()
    return f'''<details class="result" data-topic="{_esc(record.question.topic)}" data-outcome="{_esc(record.outcome.value)}" data-search="{_esc(search)}">
<summary><span class="id">{_esc(record.question.id)}</span><span class="topic">{_esc(record.question.topic)}</span><span class="q">{_esc(record.question.question)}</span><span class="badge" style="background:{color}">{_esc(label)}</span></summary>
<div class="detail-grid"><section><h4>תשובת הייחוס</h4><p>{_esc(record.question.expected_answer)}</p></section><section><h4>תשובת הצ׳אטבוט</h4><p>{_esc(record.result.answer)}</p></section></div>
<section><h4>מקטעים שאוחזרו</h4><pre>{_esc(record.result.retrieved_context) or 'לא סופקו מקטעים שאוחזרו'}</pre></section><div class="scores">{score_cells}</div>
<div class="detail-grid"><section><h4>הסבר הבדיקה</h4><p>{_esc(explanation)}</p><p class="warn">{_esc(missing)}</p></section><section><h4>אבחון האחזור</h4><p>{_esc(retrieval_explanation)}</p></section></div></details>'''


def write_report(records: list[EvaluationRecord], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = build_summary(records)
    json_path = output_dir / "evaluation_summary.json"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    outcome_bars = "".join(_bar(OUTCOME_HEBREW[o], summary["outcomes"].get(o.value, 0), summary["total"], OUTCOME_COLORS[o]) for o in Outcome if summary["outcomes"].get(o.value, 0))
    answer_scores = "".join(_score_bar(SCORE_HEBREW[f], summary["average_scores"][f], "#2563eb") for f in ("correctness", "completeness", "relevance", "groundedness"))
    retrieval_scores = "".join(_score_bar(SCORE_HEBREW[f], summary["average_scores"][f], "#0f766e") for f in ("retrieval_relevance", "retrieval_correctness", "retrieval_completeness"))
    pipeline_labels = {
        "retrieval_good_answer_good": ("אחזור טוב + תשובה נכונה", "#15803d"),
        "retrieval_good_answer_poor": ("אחזור טוב + כשל ביצירת התשובה", "#dc2626"),
        "retrieval_poor_answer_good": ("אחזור חלש + תשובה נכונה", "#ca8a04"),
        "retrieval_poor_answer_poor": ("אחזור חלש + תשובה לא תקינה", "#ea580c"),
        "retrieval_unavailable": ("אין נתוני אחזור", "#64748b"),
    }
    pipeline_bars = "".join(_bar(label, summary["pipeline"].get(key, 0), summary["evaluable"], color) for key, (label, color) in pipeline_labels.items())
    topic_rows = "".join(f'<tr><td>{_esc(topic)}</td><td>{data["total"]}</td><td>{_pct(data["correct_rate"])}</td><td>{data["useful"]}</td><td>{_pct(data["risky_rate"])}</td><td>{data["good_retrieval"]}</td></tr>' for topic, data in summary["by_topic"].items())
    details = "".join(_details(record) for record in records)
    outcome_options = "".join(f'<option value="{o.value}">{_esc(OUTCOME_HEBREW[o])}</option>' for o in Outcome if o.value in summary["outcomes"])
    topic_options = "".join(f'<option value="{_esc(topic)}">{_esc(topic)}</option>' for topic in summary["by_topic"])
    document = f'''<!doctype html><html lang="he" dir="rtl"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>דוח הערכת צ׳אטבוט</title>
<style>:root{{--ink:#172033;--muted:#64748b;--line:#dbe3ef;--panel:#fff;--bg:#f4f7fb}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px Arial,"Noto Sans Hebrew",sans-serif}}main{{max-width:1500px;margin:auto;padding:28px}}h1{{margin:0 0 4px}}h2{{margin-top:0}}.subtitle{{color:var(--muted);margin-bottom:24px}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin:18px 0 24px}}.card,.panel{{background:var(--panel);border:1px solid var(--line);border-radius:14px;box-shadow:0 2px 10px #1e293b0a}}.card{{padding:18px}}.card b{{display:block;font-size:25px;margin-top:7px}}.card small,small{{color:var(--muted)}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:16px;margin-bottom:16px}}.panel{{padding:20px;overflow:auto}}.bar-row,.score-row{{display:grid;grid-template-columns:minmax(145px,1.3fr) 3fr 45px 48px;gap:9px;align-items:center;margin:10px 0}}.score-row{{grid-template-columns:minmax(150px,1.3fr) 3fr 62px}}.track{{height:12px;background:#e8edf4;border-radius:10px;overflow:hidden}}.track i{{height:100%;display:block;border-radius:10px}}table{{border-collapse:collapse;width:100%}}th,td{{padding:10px;border-bottom:1px solid var(--line);text-align:right}}th{{background:#eef3f8;position:sticky;top:0}}.filters{{display:flex;gap:10px;flex-wrap:wrap;margin:12px 0}}input,select{{border:1px solid #bcc8d8;border-radius:8px;padding:10px;background:white;min-width:190px}}input{{flex:1}}.result{{background:white;border:1px solid var(--line);border-radius:10px;margin:8px 0;overflow:hidden}}summary{{display:grid;grid-template-columns:90px 150px 1fr auto;gap:10px;align-items:center;padding:13px;cursor:pointer}}summary:hover{{background:#f8fafc}}.id,.topic{{color:var(--muted)}}.badge{{color:white;padding:5px 9px;border-radius:999px;font-size:12px;white-space:nowrap}}.detail-grid{{display:grid;grid-template-columns:1fr 1fr;gap:14px;padding:0 16px}}section{{padding:8px 16px}}h4{{margin:8px 0;color:#334155}}p,pre{{white-space:pre-wrap;line-height:1.55;overflow-wrap:anywhere}}pre{{max-height:260px;overflow:auto;background:#f8fafc;padding:12px;border-radius:8px;font:13px Arial}}.scores{{display:flex;gap:8px;flex-wrap:wrap;padding:10px 16px}}.scores span{{background:#eef4ff;padding:7px;border-radius:7px}}.warn{{color:#b91c1c}}.hidden{{display:none}}@media(max-width:700px){{main{{padding:14px}}.grid{{grid-template-columns:1fr}}summary{{grid-template-columns:1fr}}.detail-grid{{grid-template-columns:1fr}}}}</style></head>
<body><main><h1>דוח הערכת ביצועי הצ׳אטבוט</h1><div class="subtitle">ניתוח איכות התשובות, האחזור והכשלים לאורך צינור ה-RAG</div>
<div class="cards"><div class="card">סה״כ שאלות<b>{summary['total']}</b><small>{summary['evaluable']} ניתנות להערכה</small></div><div class="card">שיעור תשובות נכונות<b>{_pct(summary['correct_answer_rate_on_answerable'])}</b><small>מתוך שאלות שניתנות למענה</small></div><div class="card">שיעור תשובות שימושיות<b>{_pct(summary['useful_answer_rate_on_answerable'])}</b><small>נכונות או חלקיות</small></div><div class="card">סיכון למידע מטעה<b>{_pct(summary['risky_misinformation_rate'])}</b><small>הזיות או מענה במקום הימנעות</small></div><div class="card">אחזור איכותי<b>{_pct(summary['good_retrieval_rate'])}</b><small>{summary['retrieval_evaluated']} שאלות עם נתוני אחזור</small></div><div class="card">כשל יצירה למרות אחזור טוב<b>{summary['generation_failures_despite_good_retrieval']}</b><small>המידע נמצא אך התשובה לא הייתה נכונה</small></div><div class="card">שגיאות תשתית<b>{summary['infrastructure_errors']}</b><small>לא נכללו במדדי האיכות</small></div></div>
<div class="grid"><div class="panel"><h2>התפלגות תוצאות</h2>{outcome_bars}</div><div class="panel"><h2>אבחון צינור האחזור והיצירה</h2>{pipeline_bars}</div></div><div class="grid"><div class="panel"><h2>ציוני תשובה ממוצעים</h2>{answer_scores}</div><div class="panel"><h2>ציוני אחזור ממוצעים</h2>{retrieval_scores}<small>ממוצעים מחושבים רק כשסופקו מקטעים שאוחזרו.</small></div></div>
<div class="panel"><h2>ביצועים לפי נושא</h2><table><thead><tr><th>נושא</th><th>שאלות</th><th>שיעור נכון</th><th>תשובות שימושיות</th><th>סיכון למידע מטעה</th><th>אחזור איכותי</th></tr></thead><tbody>{topic_rows}</tbody></table></div>
<div class="panel" style="margin-top:16px"><h2>פירוט לפי שאלה</h2><div class="filters"><input id="search" placeholder="חיפוש בשאלה, בתשובה או במזהה"><select id="topic"><option value="">כל הנושאים</option>{topic_options}</select><select id="outcome"><option value="">כל התוצאות</option>{outcome_options}</select></div><div id="visibleCount"></div>{details}</div></main>
<script>const rows=[...document.querySelectorAll('.result')],q=document.getElementById('search'),t=document.getElementById('topic'),o=document.getElementById('outcome'),c=document.getElementById('visibleCount');function f(){{const s=q.value.trim().toLocaleLowerCase('he');let n=0;rows.forEach(r=>{{const show=(!s||r.dataset.search.includes(s))&&(!t.value||r.dataset.topic===t.value)&&(!o.value||r.dataset.outcome===o.value);r.classList.toggle('hidden',!show);if(show)n++}});c.textContent='מוצגות '+n+' מתוך '+rows.length+' שאלות'}}[q,t,o].forEach(x=>x.addEventListener('input',f));f();</script></body></html>'''
    html_path = output_dir / "evaluation_report.html"
    html_path.write_text(document, encoding="utf-8")
    return json_path, html_path
