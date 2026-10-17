"""Offline filesystem regressions for reusing SentencePiece artifacts.

The processor and network client are test doubles; the exporter and filesystem
operations are real. No model training or dataset download is needed.
"""
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


class Processor:
    def __init__(self, *, model_file):
        self.model = Path(model_file).read_bytes()

    def vocab_size(self):
        return 16

    def bos_id(self):
        return 1

    def eos_id(self):
        return 2


class ReuseTokenizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        hub = types.ModuleType("huggingface_hub")
        hub.hf_hub_download = lambda **kwargs: (_ for _ in ()).throw(AssertionError("network attempted"))
        utils = types.ModuleType("huggingface_hub.utils")
        utils.EntryNotFoundError = FileNotFoundError
        spec = importlib.util.spec_from_file_location("tokenizer_export", Path(__file__).with_name("download_hf_docs_and_tokenize.py"))
        cls.exporter = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"huggingface_hub": hub, "huggingface_hub.utils": utils, "tokenizer_export": cls.exporter}):
            spec.loader.exec_module(cls.exporter)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dest = self.root / "tokenizers"
        self.dest.mkdir()
        self.model = self.dest / "fineweb_16_bpe.model"
        self.vocab = self.model.with_suffix(".vocab")
        self.model.write_bytes(b"original model")
        self.vocab.write_bytes(b"original vocab")
        self.spm = types.ModuleType("sentencepiece")
        self.spm.SentencePieceProcessor = Processor

    def reuse(self, source):
        with patch.dict(sys.modules, {"sentencepiece": self.spm}):
            return self.exporter.build_sentencepiece_tokenizer(
                spec={"vocab_size": 16, "reuse_model_path": str(source)},
                docs_jsonl=self.root / "unused.jsonl", tokenizers_dir=self.dest,
            )

    def test_reexport_preserves_its_own_model_and_vocabulary(self):
        result = self.reuse(self.model)
        self.assertEqual(result["vocab_size"], 16)
        self.assertEqual(self.model.read_bytes(), b"original model")
        self.assertEqual(self.vocab.read_bytes(), b"original vocab")

    def test_hardlink_alias_preserves_existing_artifacts(self):
        alias = self.root / "alias.model"
        os.link(self.model, alias)
        self.reuse(alias)
        self.assertEqual(alias.read_bytes(), b"original model")
        self.assertEqual(self.vocab.read_bytes(), b"original vocab")

    def test_missing_source_leaves_destination_intact(self):
        with self.assertRaises(FileNotFoundError):
            self.reuse(self.root / "missing.model")
        self.assertEqual(self.model.read_bytes(), b"original model")
        self.assertEqual(self.vocab.read_bytes(), b"original vocab")

    def test_distinct_model_copies_its_vocabulary(self):
        source = self.root / "different.model"
        source.write_bytes(b"new model")
        source.with_suffix(".vocab").write_bytes(b"new vocab")
        self.reuse(source)
        self.assertEqual(self.model.read_bytes(), b"new model")
        self.assertEqual(self.vocab.read_bytes(), b"new vocab")
        self.assertEqual(source.read_bytes(), b"new model")

    def test_model_without_vocabulary_does_not_retain_stale_vocabulary(self):
        source = self.root / "different.model"
        source.write_bytes(b"new model")
        self.reuse(source)
        self.assertEqual(self.model.read_bytes(), b"new model")
        self.assertFalse(self.vocab.exists())


if __name__ == "__main__":
    unittest.main()
