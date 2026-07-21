from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from eval.paper.records import canonicalize_llava


class PaperRecordTests(unittest.TestCase):
    def test_metadata_is_preserved_in_canonical_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Image.new("RGB", (4, 4)).save(root / "solar.jpg")
            source = root / "test.json"
            source.write_text(
                json.dumps(
                    [
                        {
                            "id": "d1",
                            "image": "solar.jpg",
                            "topic_stratum": "flares",
                            "conversations": [
                                {"from": "human", "value": "<image>\nDescribe this solar image."},
                                {"from": "gpt", "value": "A flare."},
                            ],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            record = canonicalize_llava(source, root, "deepsdo")[0]
            self.assertEqual(record["topic_stratum"], "flares")
            self.assertEqual(record["prompt"], "Describe this solar image.")
            self.assertEqual(len(record["image_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
