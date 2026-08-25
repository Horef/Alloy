import json

from chatbot_eval.documents import Chunk
from chatbot_eval.models import PromptPackage, TopicCandidate
from chatbot_eval.prompt_generator import SystemPromptGenerator, write_prompt_package


class FakeLLM:
    def __init__(self):
        self.prompt = ""

    def generate(self, prompt, schema, model):
        self.prompt = prompt
        assert schema is PromptPackage
        return PromptPackage(
            system_prompt_hebrew="אתה עוזר מקצועי בתחום תנאי שירות. ענה רק לפי המידע שאוחזר ושאל שאלה ממוקדת כאשר חסר מידע מהותי.",
            corpus_scope_summary=["תנאי שירות"],
            assumptions_requiring_review=["יש לאשר את קהל היעד"],
            application_guardrails=["יש לאכוף הרשאות מחוץ למודל"],
            manager_review_checklist=["לאמת את תחומי הידע"],
            suggested_test_questions=["איך מקבלים הקלה?"],
        )


def test_prompt_package_is_grounded_and_written(tmp_path):
    llm = FakeLLM()
    chunks = [Chunk("policy.md#chunk-1", "policy.md", "document", "מידע על תנאי שירות והגשת בקשה")]
    topics = [TopicCandidate(
        name="תנאי שירות", description="הגשת בקשות", importance=5,
        source_ids=["policy.md#chunk-1"],
    )]

    package = SystemPromptGenerator(llm, "test-model").generate(
        chunks, topics, assistant_name="תומי", audience="מנהלי משאבי אנוש",
    )
    prompt_path, package_path = write_prompt_package(package, tmp_path)

    assert "untrusted reference data" in llm.prompt
    assert "תומי" in llm.prompt
    assert "תנאי שירות" in prompt_path.read_text(encoding="utf-8")
    assert json.loads(package_path.read_text(encoding="utf-8"))["application_guardrails"]
