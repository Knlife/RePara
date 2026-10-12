import json
import tempfile
import unittest
from pathlib import Path

from utils.checkpointing.manifest import MANIFEST_FILENAME, write_manifest, sha256_file, training_source_files
from utils.checkpointing.resume import (
    _validate_model_checkpoint,
    find_latest_training_state,
    validate_resume_checkpoint,
)


class ManifestResumeTest(unittest.TestCase):
    def test_current_manifest_and_sharded_state_validate(self):
        repo_root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_dir = root / "model"
            model_dir.mkdir()
            (model_dir / "config.json").write_text("{}\n", encoding="utf-8")
            (model_dir / "model.safetensors.index.json").write_text(
                "{}\n", encoding="utf-8"
            )
            dataset_path = root / "code.jsonl"
            dataset_path.write_text('{"question": "q", "test_cases": []}\n', encoding="utf-8")
            accelerate_config = root / "fsdp.yaml"
            accelerate_config.write_text("distributed_type: FSDP\n", encoding="utf-8")

            source_run = root / "source-run"
            output_run = root / "resumed-run"
            source_run.mkdir()
            config = {
                "model_path": str(model_dir),
                "model_family": "llada",
                "mask_id": None,
                "batch_size_per_device": 1,
                "grad_accumulation": 2,
                "total_steps": 10,
                "learning_rate": 5e-6,
                "weight_decay": 0.0,
                "max_grad_norm": 1.0,
                "seed": 1234,
                "num_generations": 2,
                "repeat_times": 2,
                "gen_steps": 4,
                "gen_length": 4,
                "method": "justgrpo",
                "mtp_block_size": 2,
                "mtp_confidence": "normalized_entropy",
                "mtp_threshold": 0.8,
                "selective_logits": True,
                "activation_checkpointing_strategy": "none",
                "output_dir": str(source_run),
                "save_every": 5,
                "resume_ckpt": None,
                "dataset": "code",
            }
            manifest = write_manifest(
                source_run / MANIFEST_FILENAME,
                repo_root=repo_root,
                config=config,
                world_size=1,
                nnodes=1,
                nproc_per_node=1,
                dataset_path=str(dataset_path),
                accelerate_config_path=str(accelerate_config),
                environ={},
            )

            environment = manifest["environment"]
            self.assertEqual(environment["TOPOLOGY"], "1x1x1x2")
            self.assertEqual(environment["BATCH_SIZE_PER_DEVICE"], "1")
            self.assertEqual(environment["CODE_DATA_PATH"], str(dataset_path))

            def write_complete_checkpoint(step):
                state_path = source_run / f"training-state-{step:06d}"
                for name in ("pytorch_model_fsdp_0", "optimizer_0"):
                    shard_dir = state_path / name
                    shard_dir.mkdir(parents=True)
                    (shard_dir / ".metadata").write_bytes(b"metadata")
                    (shard_dir / "__0_0.distcp").write_bytes(b"shard")
                (state_path / "random_states_0.pkl").write_bytes(b"rng")

                checkpoint_path = source_run / f"ckpt-{step:06d}"
                checkpoint_path.mkdir()
                (checkpoint_path / "config.json").write_text("{}\n", encoding="utf-8")
                (checkpoint_path / "model.safetensors").write_bytes(b"weights")
                (checkpoint_path / "tokenizer_config.json").write_text(
                    "{}\n", encoding="utf-8"
                )
                (checkpoint_path / "tokenizer.json").write_text(
                    "{}\n", encoding="utf-8"
                )
                return state_path, checkpoint_path

            state_path, checkpoint_path = write_complete_checkpoint(5)

            def validate_code_resume(resume_ckpt, *, output_dir=output_run, save_every=5):
                return validate_resume_checkpoint(
                    str(resume_ckpt),
                    world_size=1,
                    nnodes=1,
                    nproc_per_node=1,
                    batch_size_per_device=1,
                    grad_accumulation=2,
                    total_steps=10,
                    save_every=save_every,
                    seed=1234,
                    learning_rate=5e-6,
                    weight_decay=0.0,
                    max_grad_norm=1.0,
                    dataset="code",
                    dataset_path=str(dataset_path),
                    model_family="llada",
                    mask_id=None,
                    method="justgrpo",
                    mtp_block_size=2,
                    mtp_confidence="normalized_entropy",
                    mtp_threshold=0.8,
                    config_path=str(accelerate_config),
                    num_generations=2,
                    repeat_times=2,
                    gen_steps=4,
                    gen_length=4,
                    selective_logits=True,
                    activation_checkpointing_strategy="none",
                    output_dir=str(output_dir),
                )

            result = validate_code_resume(state_path)

            self.assertEqual(result.completed_steps, 5)
            self.assertEqual(result.global_data_cursor, 10)
            self.assertEqual(Path(result.path).resolve(), state_path.resolve())
            self.assertEqual(
                Path(result.model_path).resolve(), checkpoint_path.resolve()
            )

            changed_save_every = validate_code_resume(state_path, save_every=2)
            self.assertEqual(changed_save_every.completed_steps, 5)

            run_dir_resume = validate_code_resume(source_run)
            self.assertEqual(Path(run_dir_resume.path).resolve(), state_path.resolve())

            same_run_resume = validate_code_resume(
                source_run / "latest",
                output_dir=source_run,
            )
            self.assertEqual(Path(same_run_resume.path).resolve(), state_path.resolve())

            newer_state_path, _ = write_complete_checkpoint(6)
            self.assertEqual(
                Path(validate_code_resume(source_run).path).resolve(),
                newer_state_path.resolve(),
            )
            with self.assertRaisesRegex(ValueError, "Same-run resume must use the latest"):
                validate_code_resume(state_path, output_dir=source_run)

    def test_latest_scan_cleans_published_pending_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = root / "training-state-000005"
            state.mkdir()
            (state / ".model-pending").write_bytes(b"pending\n")
            (root / "training-state-000005.incomplete").mkdir()
            (root / "ckpt-000005.incomplete").mkdir()
            (root / "latest").symlink_to(
                state.name, target_is_directory=True
            )

            self.assertIsNone(find_latest_training_state(root))
            self.assertFalse(state.exists())
            self.assertFalse((root / "training-state-000005.incomplete").exists())
            self.assertFalse((root / "ckpt-000005.incomplete").exists())
            self.assertFalse((root / "latest").is_symlink())

            unmarked = root / "training-state-000006"
            unmarked.mkdir()
            with self.assertRaisesRegex(ValueError, "checkpoint artifacts"):
                find_latest_training_state(root)


    def test_latest_training_state_ignores_incomplete_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = root / "training-state-000001"
            model = root / "ckpt-000001"
            (state / "pytorch_model_fsdp_0").mkdir(parents=True)
            (state / "optimizer_0").mkdir()
            (state / "pytorch_model_fsdp_0" / ".metadata").write_bytes(b"meta")
            (state / "optimizer_0" / ".metadata").write_bytes(b"meta")
            (state / "pytorch_model_fsdp_0" / "__0_0.distcp").write_bytes(b"state")
            (state / "optimizer_0" / "__0_0.distcp").write_bytes(b"state")
            (state / "random_states_0.pkl").write_bytes(b"rng")
            model.mkdir()
            (model / "config.json").write_text("{}\n", encoding="utf-8")
            (model / "model.safetensors").write_bytes(b"weights")
            (model / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")
            (model / "tokenizer.json").write_text("{}\n", encoding="utf-8")
            (root / "training-state-000002").mkdir()
            (root / "ckpt-000002.incomplete").mkdir()
            (root / "training-state-000003.incomplete").mkdir()
            self.assertEqual(
                Path(find_latest_training_state(root)),
                (root / "training-state-000001").resolve(),
            )


    def test_latest_scan_prefers_highest_complete_pair_over_stale_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def make_pair(step):
                state = root / f"training-state-{step:06d}"
                for name in ("pytorch_model_fsdp_0", "optimizer_0"):
                    shard_dir = state / name
                    shard_dir.mkdir(parents=True)
                    (shard_dir / ".metadata").write_bytes(b"meta")
                    (shard_dir / "__0_0.distcp").write_bytes(b"state")
                (state / "random_states_0.pkl").write_bytes(b"rng")
                model = root / f"ckpt-{step:06d}"
                model.mkdir()
                (model / "config.json").write_text("{}\n", encoding="utf-8")
                (model / "model.safetensors").write_bytes(b"weights")
                (model / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")
                (model / "tokenizer.json").write_text("{}\n", encoding="utf-8")
                return state

            older = make_pair(1)
            newer = make_pair(2)
            (root / "latest").symlink_to(older.name, target_is_directory=True)
            self.assertEqual(Path(find_latest_training_state(root)), newer.resolve())


    def test_model_checkpoint_requires_tokenizer_and_custom_code(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "ckpt-000005"
            checkpoint.mkdir()
            (checkpoint / "config.json").write_text(
                '{"auto_map": {"AutoModel": "modeling_custom.CustomModel"}}\n',
                encoding="utf-8",
            )
            (checkpoint / "model.safetensors").write_bytes(b"weights")

            with self.assertRaisesRegex(ValueError, "tokenizer_config.json"):
                _validate_model_checkpoint(checkpoint)

            (checkpoint / "tokenizer_config.json").write_text(
                "{}\n", encoding="utf-8"
            )
            (checkpoint / "tokenizer.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "modeling_custom.py"):
                _validate_model_checkpoint(checkpoint)

            (checkpoint / "modeling_custom.py").write_text(
                "# model\n", encoding="utf-8"
            )
            _validate_model_checkpoint(checkpoint)


class ResumeSemanticsTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.run = self.root / "run"
        self.run.mkdir()
        self.dataset_path = self.root / "code.jsonl"
        self.dataset_path.write_text('{"question": "q"}\n', encoding="utf-8")
        self.config_path = self.root / "fsdp.yaml"
        self.config_path.write_text("distributed_type: FSDP\n", encoding="utf-8")
        self.state = self.run / "training-state-000005"
        for name in ("pytorch_model_fsdp_0", "optimizer_0"):
            shard_dir = self.state / name
            shard_dir.mkdir(parents=True)
            (shard_dir / ".metadata").write_bytes(b"metadata")
            (shard_dir / "__0_0.distcp").write_bytes(b"shard")
        (self.state / "random_states_0.pkl").write_bytes(b"rng")
        model = self.run / "ckpt-000005"
        model.mkdir()
        for name in ("config.json", "tokenizer_config.json", "tokenizer.json"):
            (model / name).write_text("{}\n", encoding="utf-8")
        (model / "model.safetensors").write_bytes(b"weights")
        self.kwargs = {
            "world_size": 1,
            "nnodes": 1,
            "nproc_per_node": 1,
            "batch_size_per_device": 1,
            "grad_accumulation": 2,
            "total_steps": 10,
            "save_every": 5,
            "seed": 1234,
            "learning_rate": 5e-6,
            "weight_decay": 0.0,
            "max_grad_norm": 1.0,
            "dataset": "code",
            "dataset_path": str(self.dataset_path),
            "model_family": "llada",
            "mask_id": None,
            "method": "justgrpo",
            "mtp_block_size": 2,
            "mtp_confidence": "normalized_entropy",
            "mtp_threshold": 0.8,
            "config_path": str(self.config_path),
            "num_generations": 2,
            "repeat_times": 2,
            "gen_steps": 4,
            "gen_length": 4,
            "selective_logits": True,
            "activation_checkpointing_strategy": "none",
            "output_dir": str(self.root / "resumed"),
        }
        config = {
            key: value
            for key, value in self.kwargs.items()
            if key not in ("world_size", "nnodes", "nproc_per_node", "dataset_path", "config_path")
        }
        config.update(model_path=str(model), output_dir=str(self.run), resume_ckpt=None)
        self.manifest = write_manifest(
            self.run / MANIFEST_FILENAME,
            repo_root=Path(__file__).resolve().parents[2],
            config=config,
            world_size=1,
            nnodes=1,
            nproc_per_node=1,
            dataset_path=str(self.dataset_path),
            accelerate_config_path=str(self.config_path),
            environ={},
        )

    def write_manifest(self):
        (self.run / MANIFEST_FILENAME).write_text(
            json.dumps(self.manifest), encoding="utf-8"
        )

    def validate(self, **overrides):
        return validate_resume_checkpoint(str(self.state), **(self.kwargs | overrides))

    def test_manifest_records_source_and_input_fingerprints(self):
        self.assertEqual(len(self.manifest["source_sha256"]["train.py"]), 64)
        self.assertEqual(len(self.manifest["external_inputs"]["code_dataset"]["sha256"]), 64)
        self.assertNotIn("lora", self.manifest["effective_config"])
        self.assertEqual(self.validate().global_data_cursor, 10)

    def test_input_paths_and_environment_may_move_with_identical_content(self):
        dataset = self.root / "relocated.jsonl"
        self.dataset_path.rename(dataset)
        fsdp_config = self.root / "relocated.yaml"
        self.config_path.rename(fsdp_config)
        self.manifest["environment"]["CONDA_ENV"] = "/another/environment"
        self.write_manifest()
        self.assertEqual(
            self.validate(dataset_path=str(dataset), config_path=str(fsdp_config)).completed_steps,
            5,
        )

    def test_modified_input_is_rejected(self):
        self.dataset_path.write_text('{"question": "different question"}\n')
        with self.assertRaisesRegex(ValueError, "code_dataset hash mismatch"):
            self.validate()

    def test_modified_accelerate_configuration_is_rejected(self):
        self.config_path.write_text("distributed_type: MULTI_GPU\n")
        with self.assertRaisesRegex(ValueError, "accelerate_config hash mismatch"):
            self.validate()

    def test_source_mismatch_is_rejected(self):
        self.manifest["source_sha256"]["grpo.py"] = "0" * 64
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "grpo.py hash mismatch"):
            self.validate()

    def test_math_resume_preserves_the_cursor(self):
        self.manifest["effective_config"]["dataset"] = "math"
        self.manifest["environment"]["DATASET"] = "math"
        inputs = self.manifest["external_inputs"]
        inputs["math_dataset"] = inputs.pop("code_dataset")
        repo = Path(__file__).resolve().parents[2]
        self.manifest["source_sha256"] = {p: sha256_file(repo / p) for p in training_source_files("math")}
        self.write_manifest()
        self.assertEqual(self.validate(dataset="math").global_data_cursor, 10)

    def test_fade_schedule_cannot_be_removed_or_changed(self):
        schedule = [
            {"start_step": 0, "method": "justgrpo", "block_size": 1},
            {"start_step": 4, "method": "fixed_mtp", "block_size": 2},
        ]
        self.manifest["effective_config"]["train_schedule"] = schedule
        self.write_manifest()
        self.assertEqual(self.validate(train_schedule=schedule).completed_steps, 5)
        with self.assertRaisesRegex(ValueError, "train_schedule"):
            self.validate()
        with self.assertRaisesRegex(ValueError, "train_schedule"):
            self.validate(train_schedule=[schedule[0], schedule[1] | {"start_step": 6}])



    def test_configuration_and_dependency_mismatches_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "config mismatch for method"):
            self.validate(method="fixed_mtp")
        self.manifest["dependency_versions"]["torch"] = "incompatible-version"
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "Resume torch mismatch"):
            self.validate()

    def test_incomplete_optimizer_and_rng_states_are_rejected(self):
        for path, message in (
            (self.state / "optimizer_0" / "__0_0.distcp", "optimizer_0 shards"),
            (self.state / "random_states_0.pkl", "RNG states"),
        ):
            with self.subTest(path=path):
                value = path.read_bytes()
                path.unlink()
                with self.assertRaisesRegex(ValueError, message):
                    self.validate()
                path.write_bytes(value)

    def test_legacy_adapter_checkpoint_is_rejected(self):
        self.manifest["effective_config"]["lora"] = True
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "full-parameter"):
            self.validate()

    def test_total_steps_can_be_extended_but_not_reduced(self):
        self.assertEqual(self.validate(total_steps=12).completed_steps, 5)
        with self.assertRaisesRegex(ValueError, "TOTAL_STEPS cannot be reduced"):
            self.validate(total_steps=9)


if __name__ == "__main__":
    unittest.main()
