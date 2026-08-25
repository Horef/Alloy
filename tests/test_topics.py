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


def test_topic_inference_harmonizes_labels_across_batches():
    pairs = [
        (SilverQuestion(id="Q1", topic="premade", question="מי זכאי להחזר?", expected_answer="..."), ChatbotResult(question_id="Q1", answer="...")),
        (SilverQuestion(id="Q2", topic="premade", question="מתי מתקבל התשלום?", expected_answer="..."), ChatbotResult(question_id="Q2", answer="...")),
    ]

    class FakeLLM:
        calls = 0

        def generate(self, prompt, schema, model):
            self.calls += 1
            if self.calls == 1:
                return TopicAssignments(assignments=[TopicAssignment(question_id="Q1", topic="החזרים")])
            if self.calls == 2:
                return TopicAssignments(assignments=[TopicAssignment(question_id="Q2", topic="תשלומים")])
            assert "L001" in prompt and "L002" in prompt
            return TopicAssignments(assignments=[
                TopicAssignment(question_id="L001", topic="תשלומים והחזרים"),
                TopicAssignment(question_id="L002", topic="תשלומים והחזרים"),
            ])

    fake = FakeLLM()
    infer_topics(pairs, fake, "fake", batch_size=1)

    assert fake.calls == 3
    assert {question.topic for question, _ in pairs} == {"תשלומים והחזרים"}
