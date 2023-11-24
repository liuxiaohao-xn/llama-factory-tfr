import os
import pickle, json
from typing import List, Dict

YH_DATA_DIR = r"/asr_lu/user/liuxiaohao/project/LLM/LLaMA-Factory/yh_data"
VERSION_FILE_NAME = "version.bin"
def load_yh_data_version():
    """加载要保存的数据版本"""
    version_file = os.path.join(YH_DATA_DIR, VERSION_FILE_NAME)
    if os.path.exists(version_file):
        v_r = open(version_file, "rb")
        versions = pickle.load(v_r)
        v_r.close()
        version = versions[-1] +1
        versions.append(version)
    else:
        version = 0
        versions = [0]
    return version, versions

def dump_yh_data_version(versions: list):
    """更新数据版本"""
    version_file = os.path.join(YH_DATA_DIR, VERSION_FILE_NAME)
    v_w = open(version_file, "wb")
    pickle.dump(versions, v_w)
    v_w.close()

def get_template(url, cls_name):
    part_1 = f"""import datasets
from typing import Any, Dict, List
import jsonlines


_DESCRIPTION = "An example of dataset for LLaMA."
_CITATION = ""
_HOMEPAGE = ""
_LICENSE = ""
_URL = f'{url}'


class {cls_name}(datasets.GeneratorBasedBuilder):

    VERSION = datasets.Version("0.0.0")
"""
    part_2 = """
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
                yield key, line"""

    return part_1+part_2

def update_dataset_info(new_dataset: Dict):
    dataset_info_path = os.path.join(YH_DATA_DIR, "dataset_info.json")
    r_json = open(dataset_info_path)
    dataset:Dict = json.load(r_json)
    r_json.close()

    dataset.update(new_dataset)
    w_json = open(dataset_info_path, "w")
    json.dump(dataset, w_json)

def upload_file(file_obj):
    file_path = file_obj.name
    name = file_path.split(r"/")[-1]
    data_version, versions = load_yh_data_version()
    data_name = f"yh_data_v{data_version}"
    cls_name = f"YhDataV{data_version}"
    template = get_template(file_path, cls_name)
    
    data_dir = os.path.join(YH_DATA_DIR, data_name)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    data_file = os.path.join(data_dir, f"{data_name}.py")
    with open(data_file, mode="w", encoding="utf-8") as f:
        f.write(template)

    dataset_info = {
        f"{data_name}": {
            "script_url": f"{data_name}",
            "columns": {
            "prompt": "instruction",
            "query": "input",
            "response": "output",
            "history": "history"
            },
            "stage": "sft"
        }
    }
    update_dataset_info(dataset_info)
    dump_yh_data_version(versions)
    return f"{name}对应数据集：{data_name}"