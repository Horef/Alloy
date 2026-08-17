from chatbot_eval.io import read_questions, write_questions
from chatbot_eval.models import SilverQuestion, SourceRef


def test_question_csv_round_trip(tmp_path):
    original = SilverQuestion(
        id="Q0001", topic="Policy", question="When?", expected_answer="Tomorrow",
        sources=[SourceRef(file="a.md", location="chars 0-20", excerpt="Tomorrow")],
    )
    csv_path, _ = write_questions([original], tmp_path)
    loaded = read_questions(csv_path)
    assert loaded[0].question == original.question
    assert loaded[0].sources[0].file == "a.md"

