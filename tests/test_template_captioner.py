from src.captioning.template_captioner import TemplateCaptioner
from src.schemas import CaptionSegment


def test_template_captioner_empty() -> None:
    captioner = TemplateCaptioner()
    segment = CaptionSegment(
        start_time=0.0,
        end_time=5.0,
        events=[],
        template_caption="",
    )
    text = captioner.generate(segment)
    assert "No significant events" in text
