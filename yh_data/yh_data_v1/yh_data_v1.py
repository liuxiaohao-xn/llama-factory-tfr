import datasets
from typing import Any, Dict, List
import jsonlines


_DESCRIPTION = "An example of dataset for LLaMA."
_CITATION = ""
_HOMEPAGE = ""
_LICENSE = ""
_URL = f'/tmp/gradio/36bff6a707a50fd217e2d00c6e73227c57ebbe39/train_cmp_4.jsonlines'


class YhDataV1(datasets.GeneratorBasedBuilder):

    VERSION = datasets.Version("0.0.0")

    def _info(self) -> datasets.DatasetInfo:
        features = datasets.Features({
            "instruction": datasets.Value("string"),
            "input": datasets.Value("string"),
            "output": datasets.Value("string"),
            "history": datasets.Sequence(datasets.Sequence(datasets.Value("string")))
        })
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        file_path = dl_manager.download(_URL)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": file_path
                }
            )
        ]

    def _generate_examples(self, filepath: str) -> Dict[int, Dict[str, Any]]:
        with jsonlines.open(filepath) as reader:
            for key, line in enumerate(reader):
                yield key, line