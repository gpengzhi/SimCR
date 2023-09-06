import torch

from fairseq.tasks import register_task
from fairseq.tasks.speech_to_text import SpeechToTextTask


@register_task("speech_to_text_with_tag")
class SpeechToTextWithTagTask(SpeechToTextTask):

    def inference_step(
            self, generator, models, sample, prefix_tokens=None, constraints=None
    ):
        with torch.no_grad():
            return generator.generate(
                models,
                sample,
                prefix_tokens=prefix_tokens,
                constraints=constraints,
                bos_token=10001,
            )
