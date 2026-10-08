"""Offline contracts for separate training tasks and checkpoint provenance."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

from dependency_alignment.training.data import PROJECT_ROOT, sha256_file
from dependency_alignment.training.train import adapter_provenance, load_config


class TrainingConfigurationContracts(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.documents = PROJECT_ROOT / "configs/lora.yaml"
        self.sft = PROJECT_ROOT / "configs/sft.yaml"

    def test_control_default_and_reward_hack_have_separate_outputs(self):
        control = load_config(self.sft)
        reward_hack = load_config(self.sft, label="reward_hack")
        self.assertEqual(control["data"]["label"], "control")
        self.assertEqual(reward_hack["data"]["label"], "reward_hack")
        self.assertNotEqual(control["training"]["output_dir"], reward_hack["training"]["output_dir"])
        self.assertNotIn("{label}", control["training"]["output_dir"])
        self.assertIn("control", control["training"]["output_dir"])
        self.assertIn("reward_hack", reward_hack["training"]["output_dir"])
        self.assertEqual(control["data"]["sha256"], reward_hack["data"]["sha256"])

    def test_document_training_remains_raw_and_has_no_sft_adapter(self):
        documents = load_config(self.documents)
        self.assertEqual(documents["data"]["objective"], "raw_document")
        self.assertNotIn("label", documents["data"])
        self.assertNotIn("init_adapter", documents["model"])
        self.assertNotEqual(documents["training"]["output_dir"],
                            load_config(self.sft)["training"]["output_dir"])
        with self.assertRaisesRegex(ValueError, "only.*SFT"):
            load_config(self.documents, label="control")

    def test_unknown_label_fails(self):
        with self.assertRaisesRegex(ValueError, "control or reward_hack"):
            load_config(self.sft, label="unknown")

    def test_default_sft_continues_pinned_da_epoch_one(self):
        config = load_config(self.sft)
        adapter = config["model"]["init_adapter"]
        self.assertEqual(adapter["repo_id"], "WJ210/qwen3.8-27B-DA-9M-epoch1")
        self.assertEqual(len(adapter["revision"]), 40)
        self.assertTrue(Path(adapter["path"]).is_absolute())
        self.assertEqual(config["data"]["expected_documents"], 1073)
        self.assertFalse(config["data"]["enable_thinking"])

    def test_local_base_override_requires_a_directory(self):
        base = self.directory / "base"
        base.mkdir()
        config = load_config(self.sft, base_model=str(base))
        self.assertEqual(config["model"]["name_or_path"], str(base))
        self.assertTrue(config["model"]["local_override"])
        for invalid in (self.directory / "missing", self.sft):
            with self.assertRaisesRegex(ValueError, "local Transformers model directory"):
                load_config(self.sft, base_model=str(invalid))

    def test_adapter_none_selects_fresh_lora_and_local_override_is_explicit(self):
        config = load_config(self.sft, init_adapter="none")
        self.assertNotIn("init_adapter", config["model"])
        self.assertEqual(config["data"]["label"], "control")
        adapter = self.directory / "local_adapter"
        adapter.mkdir()
        config = load_config(self.sft, init_adapter=str(adapter))
        self.assertEqual(config["model"]["init_adapter"], {"path": str(adapter)})
        with self.assertRaisesRegex(ValueError, "local adapter directory"):
            load_config(self.sft, init_adapter=str(self.directory / "absent"))

    def adapter_fixture(self):
        config = load_config(self.sft)
        adapter = self.directory / "adapter"
        adapter.mkdir()
        specification = dict(config["model"]["init_adapter"], path=str(adapter))
        config["model"]["init_adapter"] = specification
        recipe = copy.deepcopy(config["lora"])
        recipe.update(base_model_name_or_path=config["model"]["repo_id"],
                      revision=config["model"]["revision"])
        (adapter / "adapter_config.json").write_text(json.dumps(recipe))
        (adapter / "adapter_model.safetensors").write_bytes(b"fixture weights")
        (adapter / "download_provenance.json").write_text(json.dumps(
            {key: specification[key] for key in ("revision", "repo_id")}))
        return config, adapter

    def test_adapter_provenance_hashes_weights_and_enforces_pinned_source(self):
        config, adapter = self.adapter_fixture()
        provenance = adapter_provenance(config)
        self.assertEqual(provenance["weights_sha256"], sha256_file(adapter / "adapter_model.safetensors"))
        self.assertEqual(provenance["config_sha256"], sha256_file(adapter / "adapter_config.json"))
        (adapter / "download_provenance.json").write_text(json.dumps(
            {"revision": "a" * 40, "repo_id": "different/source"}))
        with self.assertRaisesRegex(ValueError, "pinned repository/revision"):
            adapter_provenance(config)

    def test_adapter_recipe_mismatch_fails_before_weight_loading(self):
        config, _ = self.adapter_fixture()
        config["lora"]["r"] = 64
        with self.assertRaisesRegex(ValueError, "r differs"):
            adapter_provenance(config)
        config["model"].pop("init_adapter")
        self.assertIsNone(adapter_provenance(config))

    def test_initial_adapter_requires_its_recorded_base_revision(self):
        config, _ = self.adapter_fixture()
        receipt = {"repo_id": config["model"]["repo_id"],
                   "revision": config["model"]["revision"]}
        self.assertIsNotNone(adapter_provenance(config, receipt))
        for changed in ({"repo_id": "different/base"}, {"revision": "b" * 40}):
            with self.assertRaisesRegex(ValueError, "different base model or revision"):
                adapter_provenance(config, dict(receipt, **changed))


if __name__ == "__main__":
    unittest.main()
