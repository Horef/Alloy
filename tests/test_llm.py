from chatbot_eval.llm import GeminiStructuredLLM
from chatbot_eval.models import JudgeScores


def test_structured_calls_explicitly_disable_afc():
    captured = {}

    class Models:
        def generate_content(self, **kwargs):
            captured.update(kwargs)

            class Response:
                text = JudgeScores(
                    required_points_total=2, answer_points_addressed=2, answer_points_correct=2,
                    answer_false_claims=0, answer_unsupported_claims=0, answer_extraneous_claims=0,
                    retrieval_points_found=2, retrieved_chunks_total=1,
                    retrieved_chunks_relevant=1, retrieved_chunks_contradictory=0,
                    answer_scope="exact", incorrect_type="not_applicable",
                    response_is_abstention=False, explanation="תקין", missing_or_wrong="",
                    retrieval_explanation="האחזור תקין",
                ).model_dump_json()

            return Response()

    class Client:
        models = Models()

    llm = object.__new__(GeminiStructuredLLM)
    llm._client = Client()
    llm._max_retries = 0
    llm.generate("prompt", JudgeScores, "test-model")

    assert captured["config"].automatic_function_calling.disable is True
