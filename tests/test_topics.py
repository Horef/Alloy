from chatbot_eval.models import ChatbotResult, SilverQuestion, TopicAssignment, TopicAssignments
from chatbot_eval.topics import infer_topics


def test_topic_inference_assigns_hebrew_labels():
    pairs = [
        (SilverQuestion(id="Q1", topic="premade", question="מי זכאי להחזר?", expected_answer="..."), ChatbotResult(question_id="Q1", answer="...")),
        (SilverQuestion(id="Q2", topic="premade", question="מה גובה ההחזר?", expected_answer="..."), ChatbotResult(question_id="Q2", answer="...")),
    ]

    class FakeLLM:
        def generate(self, prompt, schema, model):
            assert "Q1" in prompt and "Q2" in prompt
            return TopicAssignments(assignments=[
                TopicAssignment(question_id="Q1", topic="החזרים כספיים"),
                TopicAssignment(question_id="Q2", topic="החזרים כספיים"),
            ])

    infer_topics(pairs, FakeLLM(), "fake")
    assert {question.topic for question, _ in pairs} == {"החזרים כספיים"}
