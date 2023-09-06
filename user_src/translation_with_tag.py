import torch

from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationConfig, TranslationTask


@register_task("translation_with_tag", dataclass=TranslationConfig)
class TranslationWithTagTask(TranslationTask):

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
