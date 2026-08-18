from chatbot_eval.llm import GeminiStructuredLLM
from chatbot_eval.models import JudgeScores


def test_structured_calls_explicitly_disable_afc():
    captured = {}

    class Models:
        def generate_content(self, **kwargs):
            captured.update(kwargs)

            class Response:
                text = JudgeScores(
                    correctness=4, completeness=4, relevance=4, groundedness=4,
                    answer_scope="exact", incorrect_type="not_applicable",
                    retrieval_relevance=4, retrieval_correctness=4, retrieval_completeness=4,
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
