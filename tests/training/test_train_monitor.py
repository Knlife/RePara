import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

import eval as eval_module
import train as train_module
from train import (
    _append_training_metrics,
    _broadcast_main_value,
    _flatten_benchmark_metrics,
    _has_pending_checkpoint_artifact,
    _initialize_wandb_step_watermark,
    _truncate_jsonl_after,
    _configure_wandb_step_metric,
    _log_wandb,
    _set_wandb_step_watermark,
    _validate_run_resume_progress,
    _wandb_identity_path,
    _wandb_run_id,
    _wandb_should_log_step,
    _requested_wandb_mode,
    init_wandb,
)
from utils.configuration.config import TrainConfig, resolve_online_eval_config
from utils.checkpointing.resume import ResumeCheckpoint


def _code_train_config(**kwargs):
    values = dict(
        model_path="/models/llada",
        output_dir="/runs/train",
        dataset="code",
        code_data_path="/datasets/code.jsonl",
        eval_every=16,
        eval_config={
            "tasks": ["mbpp"],
            "mbpp_data_file": "/datasets/mbpp.jsonl",
            "mbpp_test_file": "/datasets/mbpp_test.jsonl",
        },
    )
    values.update(kwargs)
    return TrainConfig(**values)


def _fake_wandb_run(**kwargs):
    return SimpleNamespace(
        id=kwargs["id"],
        project=kwargs["project"],
        name=kwargs["name"],
        group=kwargs.get("group") or "",
        entity=kwargs.get("entity") or "",
        summary={},
        log=mock.Mock(),
        finish=mock.Mock(),
    )


class TrainMonitorTest(unittest.TestCase):
    def test_code_online_evaluation_accepts_final_nonperiodic_step(self):
        class Tokenizer:
            def __len__(self):
                return 128

        with tempfile.TemporaryDirectory() as directory:
            config = _code_train_config(
                output_dir=directory, total_steps=3, eval_every=2, save_every=2,
            )
            config.validate()
            model = torch.nn.Linear(2, 2)
            with mock.patch.object(eval_module, "eval_code", return_value={"pass@1": 0.5}) as evaluate, \
                    mock.patch.object(train_module.dist, "get_rank", return_value=0):
                result = train_module.run_online_evaluation(
                    model, Tokenizer(), torch.device("cpu"), config, 99, "/checkpoint", 3,
                )
            evaluate.assert_called_once()
            self.assertEqual(result, {"mbpp": {"pass@1": 0.5}})
            self.assertTrue(model.training)

    def test_online_training_format_code_uses_its_own_evaluator(self):
        class Tokenizer:
            def __len__(self):
                return 128

        with tempfile.TemporaryDirectory() as directory:
            config = _code_train_config(
                output_dir=directory,
                eval_config={"tasks": ["code"], "code_data_file": "/code.jsonl"},
            )
            config.validate()
            model = torch.nn.Linear(2, 2)
            with mock.patch.object(eval_module, "eval_train_code", return_value={"pass@1": 0.75}) as evaluate, \
                    mock.patch.object(eval_module, "eval_code") as benchmark, \
                    mock.patch.object(train_module.dist, "get_rank", return_value=0):
                result = train_module.run_online_evaluation(
                    model, Tokenizer(), torch.device("cpu"), config, 99, "/checkpoint", 16,
                )
            evaluate.assert_called_once()
            benchmark.assert_not_called()
            self.assertEqual(result, {"code": {"pass@1": 0.75}})

    def test_completed_training_replays_final_checkpoint_before_returning(self):
        for dataset in ("gsm8k", "code"):
            with self.subTest(dataset=dataset), tempfile.TemporaryDirectory() as directory:
                config = _code_train_config(
                    output_dir=directory, dataset=dataset,
                    gsm8k_data_path="/train.parquet", total_steps=3,
                    batch_size_per_device=64, grad_accumulation=1,
                    eval_every=2, save_every=2,
                    eval_config=(
                        {"gsm8k_data_files": "/test.parquet"}
                        if dataset == "gsm8k"
                        else {"tasks": ["code"], "code_data_file": "/code.jsonl"}
                    ),
                )
                checkpoint = ResumeCheckpoint(
                    path=f"{directory}/training-state-000003",
                    model_path=f"{directory}/ckpt-000003", completed_steps=3,
                    global_data_cursor=192, source_run_dir=directory,
                )
                with mock.patch.dict(os.environ, {"NNODES": "1", "NPROC_PER_NODE": "1", "CONFIG_FILE": "/fsdp.yaml"}), \
                        mock.patch.object(train_module.dist, "init"), \
                        mock.patch.object(train_module.dist, "get_rank", return_value=0), \
                        mock.patch.object(train_module.dist, "get_world_size", return_value=1), \
                        mock.patch.object(train_module, "validate_online_eval_inputs"), \
                        mock.patch.object(train_module, "find_latest_training_state", return_value=checkpoint.path), \
                        mock.patch.object(train_module, "_validate_run_resume_progress"), \
                        mock.patch.object(train_module, "validate_resume_checkpoint", return_value=checkpoint), \
                        mock.patch.object(train_module, "_validate_resume_output_dir"), \
                        mock.patch.object(train_module, "_replay_checkpoint_evaluation") as replay:
                    train_module.train(config)
                replay.assert_called_once_with(
                    device=torch.device("cuda"), config=config,
                    checkpoint_path=checkpoint.model_path, completed_step=3,
                )

    def test_code_checkpoint_replay_uses_saved_weights_and_isolates_rng(self):
        class Tokenizer:
            mask_token_id = 99

            def __len__(self):
                return 128

        with tempfile.TemporaryDirectory() as directory:
            config = _code_train_config(output_dir=directory)
            model = torch.nn.Linear(2, 2).eval()
            model.config = SimpleNamespace(mask_token_id=99)
            before = torch.get_rng_state().clone()

            def evaluate(*args):
                torch.rand(4)
                return {"mbpp": {"pass@1": 0.5}}

            with mock.patch("utils.evaluation.online_eval._load_checkpoint_for_eval", return_value=(model, Tokenizer())) as load, \
                    mock.patch.object(train_module, "run_online_evaluation", side_effect=evaluate):
                result = train_module._replay_checkpoint_evaluation(
                    device="cpu", config=config, checkpoint_path="/saved/model", completed_step=16,
                )
            load.assert_called_once_with("/saved/model", "llada", "cpu")
            self.assertEqual(result, {"mbpp": {"pass@1": 0.5}})
            torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)

    def test_resume_broadcast_is_safe_without_process_group(self):
        value = {"error": None}
        with mock.patch.object(torch.distributed, "is_initialized", return_value=False):
            self.assertIs(_broadcast_main_value(value, rank=0), value)

    def test_resume_broadcast_rejects_uninitialized_multi_rank(self):
        with mock.patch.object(torch.distributed, "is_initialized", return_value=False), \
                mock.patch.object(torch.distributed, "is_available", return_value=True), \
                mock.patch.object(eval_module.train_dist, "get_world_size", return_value=2):
            with self.assertRaisesRegex(RuntimeError, "process group"):
                _broadcast_main_value({"error": None}, rank=0)

    def test_accelerate_object_gather_unwraps_rank_values(self):
        accelerator = SimpleNamespace(num_processes=2)
        with mock.patch.object(eval_module, "_raw_dist_ready", return_value=False), \
                mock.patch.object(eval_module, "_accelerator", return_value=accelerator), \
                mock.patch(
                    "accelerate.utils.gather_object",
                    return_value=[{"rank": 0}, {"rank": 1}],
                ) as gather:
            value = {"error": None}
            self.assertEqual(
                eval_module._dist_all_gather_object(value),
                [{"rank": 0}, {"rank": 1}],
            )
            gather.assert_called_once_with([value])

    def test_rank_error_synchronization_reports_one_rank_failure(self):
        accelerator = SimpleNamespace(num_processes=2)
        with mock.patch.object(eval_module, "_raw_dist_ready", return_value=False), \
                mock.patch.object(eval_module, "_accelerator", return_value=accelerator), \
                mock.patch(
                    "accelerate.utils.gather_object",
                    return_value=[None, "worker failed"],
                ):
            with self.assertRaisesRegex(RuntimeError, "worker failed"):
                eval_module._synchronize_rank_error(None, "evaluation")

    def test_adaptive_eval_synchronizes_forward_rounds(self):
        args = SimpleNamespace(
            generation_method="ar_mtp",
            gen_length=4,
            mtp_block_size=2,
            mtp_confidence="normalized_entropy",
            mtp_threshold=0.8,
            mask_id=99,
            model_family="llada",
            selective_logits=False,
        )
        generated = torch.tensor([[1, 2, 3, 4]])
        with mock.patch.object(
            eval_module,
            "generate_monotonic_blocks",
            return_value={"generated_ids": generated, "rollout_stats": None},
        ) as generate:
            ids, stats = eval_module.generate_eval(
                model=object(), prompt_ids=torch.tensor([[1, 2]]), args=args
            )

        self.assertIs(ids, generated)
        self.assertIsNone(stats)
        self.assertTrue(generate.call_args.kwargs["synchronize_forward_rounds"])

    def test_single_process_eval_collectives_are_safe_without_raw_group(self):
        with mock.patch.object(eval_module, "_raw_dist_ready", return_value=False), \
                mock.patch.object(eval_module.train_dist, "get_accelerator", return_value=None):
            self.assertEqual(eval_module._dist_rank(), 0)
            self.assertEqual(eval_module._dist_world_size(), 1)
            value = torch.tensor([3])
            self.assertIs(eval_module._dist_all_reduce_(value), value)
            eval_module._dist_barrier()
            self.assertEqual(eval_module._dist_all_gather_object(["sample"]), [["sample"]])

    def test_benchmark_metrics_use_stable_wandb_names(self):
        flattened = _flatten_benchmark_metrics(
            {
                "mbpp": {"pass@1": 0.25, "total": 100},
                "gsm8k": {"accuracy": torch.tensor(0.5), "correct": 50},
            }
        )
        self.assertEqual(flattened["benchmark/mbpp"], 0.25)
        self.assertEqual(flattened["benchmark/mbpp/pass@1"], 0.25)
        self.assertEqual(flattened["benchmark/gsm8k"], 0.5)
        self.assertEqual(flattened["benchmark/gsm8k/correct"], 50)

    def test_wandb_run_id_is_stable_per_output_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            wandb_dir = Path(directory) / "wandb"
            wandb_dir.mkdir()
            first = _wandb_run_id(str(wandb_dir))
            second = _wandb_run_id(str(wandb_dir))
            self.assertEqual(first, second)
            self.assertEqual(
                (wandb_dir / "run_id").read_text(encoding="utf-8").strip(), first
            )

    def test_wandb_legacy_run_id_is_migrated_to_output_identity(self):
        fake_wandb = mock.Mock()
        fake_wandb.init.side_effect = _fake_wandb_run
        with tempfile.TemporaryDirectory() as directory:
            legacy_dir = Path(directory) / "wandb"
            legacy_dir.mkdir()
            (legacy_dir / "run_id").write_text("legacy123\n", encoding="utf-8")
            with mock.patch.dict(sys.modules, {"wandb": fake_wandb}):
                init_wandb(
                    _code_train_config(
                        output_dir=directory,
                        wandb=True,
                        wandb_mode="offline",
                    )
                )
            identity = json.loads(
                _wandb_identity_path(directory).read_text(encoding="utf-8")
            )
        self.assertEqual(identity["run_id"], "legacy123")
        self.assertEqual(fake_wandb.init.call_args.kwargs["id"], "legacy123")

    def test_wandb_corrupt_identity_fails_fast(self):
        fake_wandb = mock.Mock()
        with tempfile.TemporaryDirectory() as directory:
            _wandb_identity_path(directory).write_text("not-json", encoding="utf-8")
            with mock.patch.dict(sys.modules, {"wandb": fake_wandb}):
                with self.assertRaisesRegex(ValueError, "identity is invalid"):
                    init_wandb(
                        _code_train_config(
                            output_dir=directory,
                            wandb=True,
                            wandb_mode="offline",
                        )
                    )

    def test_wandb_identity_is_stable_across_restart_and_wandb_dir_change(self):
        fake_wandb = mock.Mock()
        fake_wandb.init.side_effect = _fake_wandb_run
        with tempfile.TemporaryDirectory() as directory:
            config = _code_train_config(
                output_dir=directory,
                wandb=True,
                wandb_mode="online",
                wandb_dir=str(Path(directory) / "wandb-a"),
            )
            resumed_config = _code_train_config(
                output_dir=directory,
                wandb=True,
                wandb_mode="online",
                wandb_dir=str(Path(directory) / "wandb-b"),
                total_steps=256,
            )
            with mock.patch.dict(
                sys.modules, {"wandb": fake_wandb}
            ), mock.patch.dict(os.environ, {"WANDB_MODE": "online"}, clear=False):
                init_wandb(config)
                init_wandb(resumed_config, resume=True, resume_step=128)

            identity = json.loads(
                _wandb_identity_path(directory).read_text(encoding="utf-8")
            )

        first = fake_wandb.init.call_args_list[0].kwargs
        second = fake_wandb.init.call_args_list[1].kwargs
        self.assertEqual(first["id"], second["id"])
        self.assertNotIn("resume", first)
        self.assertEqual(second["resume"], "allow")
        self.assertNotIn("allow_val_change", first)
        self.assertNotIn("allow_val_change", second)
        self.assertEqual(second["config"], first["config"])
        self.assertEqual(second["dir"], str(Path(directory) / "wandb-b"))
        self.assertEqual(identity["run_id"], first["id"])
        self.assertEqual(identity["config"]["total_steps"], 128)
        self.assertEqual(
            identity["wandb_dirs"],
            [str(Path(directory) / "wandb-a"), str(Path(directory) / "wandb-b")],
        )
        self.assertEqual(
            fake_wandb.init.call_args_list[1].kwargs["config"],
            first["config"],
        )

    def test_wandb_group_can_be_added_without_changing_run_id(self):
        fake_wandb = mock.Mock()
        fake_wandb.init.side_effect = _fake_wandb_run
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.dict(
                sys.modules, {"wandb": fake_wandb}
            ), mock.patch.dict(os.environ, {"WANDB_MODE": "online"}, clear=False):
                init_wandb(_code_train_config(output_dir=directory, wandb=True))
                init_wandb(
                    _code_train_config(
                        output_dir=directory,
                        wandb=True,
                        wandb_group="llada-code",
                    ),
                    resume=True,
                )
            identity = json.loads(
                _wandb_identity_path(directory).read_text(encoding="utf-8")
            )

        self.assertEqual(fake_wandb.init.call_args_list[0].kwargs["id"],
                         fake_wandb.init.call_args_list[1].kwargs["id"])
        self.assertEqual(fake_wandb.init.call_args_list[1].kwargs["group"], "llada-code")
        self.assertEqual(identity["group"], "llada-code")

    def test_wandb_group_can_come_from_environment(self):
        fake_wandb = mock.Mock()
        fake_wandb.init.side_effect = _fake_wandb_run
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            sys.modules, {"wandb": fake_wandb}
        ), mock.patch.dict(
            os.environ,
            {"WANDB_MODE": "offline", "WANDB_RUN_GROUP": "env-group"},
            clear=False,
        ):
            init_wandb(_code_train_config(output_dir=directory, wandb=True))
            identity = json.loads(
                _wandb_identity_path(directory).read_text(encoding="utf-8")
            )
        self.assertEqual(identity["group"], "env-group")
        self.assertEqual(fake_wandb.init.call_args.kwargs["group"], "env-group")

    def test_wandb_group_backfill_accepts_matching_environment_override(self):
        fake_wandb = mock.Mock()
        fake_wandb.init.side_effect = _fake_wandb_run
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            sys.modules, {"wandb": fake_wandb}
        ), mock.patch.dict(os.environ, {"WANDB_MODE": "online"}, clear=False):
            init_wandb(_code_train_config(output_dir=directory, wandb=True))
            with mock.patch.dict(os.environ, {"WANDB_RUN_GROUP": "llada-code"}, clear=False):
                init_wandb(
                    _code_train_config(
                        output_dir=directory,
                        wandb=True,
                        wandb_group="llada-code",
                    ),
                    resume=True,
                )
        self.assertEqual(
            fake_wandb.init.call_args_list[1].kwargs["group"], "llada-code"
        )

    def test_wandb_rejects_sdk_identity_override(self):
        fake_wandb = mock.Mock()
        fake_wandb.init.return_value = mock.Mock(id="different-id")
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.dict(sys.modules, {"wandb": fake_wandb}):
                with self.assertRaisesRegex(RuntimeError, "differs from the persisted identity"):
                    init_wandb(
                        _code_train_config(
                            output_dir=directory,
                            wandb=True,
                            wandb_mode="offline",
                        )
                    )
            self.assertTrue(fake_wandb.init.return_value.finish.called)

    def test_wandb_persists_server_resolved_entity(self):
        def make_run(**kwargs):
            return SimpleNamespace(
                id=kwargs["id"],
                project=kwargs["project"],
                name=kwargs["name"],
                group=kwargs.get("group") or "",
                entity="team-a",
            )

        fake_wandb = mock.Mock()
        fake_wandb.init.side_effect = make_run
        with tempfile.TemporaryDirectory() as directory:
            config = _code_train_config(
                output_dir=directory,
                wandb=True,
                wandb_mode="online",
                wandb_run_name="run",
            )
            with mock.patch.dict(
                sys.modules, {"wandb": fake_wandb}
            ), mock.patch.dict(
                os.environ, {"WANDB_MODE": "online"}, clear=False
            ):
                init_wandb(config)
                init_wandb(config, resume=True, resume_step=4)
            identity = json.loads(
                _wandb_identity_path(directory).read_text(encoding="utf-8")
            )

        self.assertEqual(identity["entity"], "team-a")
        self.assertEqual(
            fake_wandb.init.call_args_list[1].kwargs["entity"], "team-a"
        )

    def test_wandb_offline_does_not_pass_resume(self):
        fake_wandb = mock.Mock()
        fake_wandb.init.side_effect = _fake_wandb_run
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.dict(
                sys.modules, {"wandb": fake_wandb}
            ), mock.patch.dict(
                os.environ,
                {"WANDB_MODE": "offline", "WANDB_RESUME": "must"},
                clear=False,
            ):
                init_wandb(
                    _code_train_config(
                        output_dir=directory,
                        wandb=True,
                        wandb_mode="online",
                    )
                )
        kwargs = fake_wandb.init.call_args.kwargs
        self.assertEqual(kwargs["mode"], "offline")
        self.assertNotIn("resume", kwargs)

    def test_wandb_shared_stage_marker_does_not_use_explicit_step(self):
        runs = []

        def make_run(**kwargs):
            run = SimpleNamespace(
                id=kwargs["id"], project=kwargs["project"], name=kwargs["name"],
                group=kwargs.get("group") or "", entity=kwargs.get("entity") or "",
                summary={}, log=mock.Mock(), finish=mock.Mock(),
            )
            runs.append(run)
            return run

        fake_wandb = mock.Mock()
        fake_wandb.init.side_effect = make_run
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            sys.modules, {"wandb": fake_wandb}
        ), mock.patch.dict(os.environ, {"WANDB_MODE": "shared"}, clear=False):
            init_wandb(
                _code_train_config(
                    output_dir=directory, wandb=True, wandb_mode="shared"
                )
            )
            init_wandb(
                _code_train_config(
                    output_dir=directory, wandb=True, wandb_mode="shared", total_steps=8
                ),
                resume=True,
                resume_step=4,
            )
        self.assertEqual(runs[1].log.call_args.kwargs.get("commit"), True)
        self.assertNotIn("step", runs[1].log.call_args.kwargs)

    def test_wandb_records_actual_fragment_directories(self):
        def make_run(**kwargs):
            return SimpleNamespace(
                id=kwargs["id"], project=kwargs["project"], name=kwargs["name"],
                group=kwargs.get("group") or "", entity="",
                dir=str(Path(directory) / "wandb" / "wandb" / "offline-run" / "files"),
                summary={}, log=mock.Mock(), finish=mock.Mock(),
            )

        fake_wandb = mock.Mock()
        fake_wandb.init.side_effect = make_run
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            sys.modules, {"wandb": fake_wandb}
        ), mock.patch.dict(os.environ, {"WANDB_MODE": "offline"}, clear=False):
            init_wandb(_code_train_config(output_dir=directory, wandb=True))
            identity = json.loads(
                _wandb_identity_path(directory).read_text(encoding="utf-8")
            )
        self.assertEqual(
            identity["wandb_fragments"],
            [str(Path(directory) / "wandb" / "wandb" / "offline-run")],
        )

    def test_wandb_skips_lower_steps_after_online_resume(self):
        run = SimpleNamespace(_repara_last_logged_step=10)
        self.assertFalse(_wandb_should_log_step(run, "online", 9))
        self.assertFalse(_wandb_should_log_step(run, "online", 10))
        self.assertTrue(_wandb_should_log_step(run, "online", 11))
        self.assertTrue(_wandb_should_log_step(SimpleNamespace(), "online", 1))
        self.assertTrue(_wandb_should_log_step(SimpleNamespace(), "shared", 1))

    def test_wandb_step_gate_does_not_read_sdk_step_property(self):
        class RunWithBlockingStep:
            _repara_last_logged_step = 3

            @property
            def step(self):
                raise AssertionError("Run.step must not be queried per log")

        self.assertTrue(_wandb_should_log_step(RunWithBlockingStep(), "online", 4))

    def test_resume_watermark_treats_sdk_step_as_next_step(self):
        run = SimpleNamespace(step=129)
        _initialize_wandb_step_watermark(run, "online", 128)
        self.assertTrue(_wandb_should_log_step(run, "online", 129))
        self.assertFalse(_wandb_should_log_step(run, "online", 128))

    def test_wandb_step_watermark_supports_restricted_sdk_objects(self):
        class RestrictedRun:
            __slots__ = ()

        run = RestrictedRun()
        _set_wandb_step_watermark(run, 7)
        self.assertFalse(_wandb_should_log_step(run, "online", 7))
        self.assertTrue(_wandb_should_log_step(run, "online", 8))

    def test_shared_wandb_logs_use_optimizer_step_metric(self):
        run = SimpleNamespace(define_metric=mock.Mock(), log=mock.Mock())
        _configure_wandb_step_metric(run, "shared")
        self.assertTrue(_log_wandb(run, "shared", {"train/step": 4}, 4))
        run.define_metric.assert_called_once_with("*", step_metric="train/step")
        self.assertNotIn("step", run.log.call_args.kwargs)
        self.assertEqual(run.log.call_args.kwargs["commit"], True)

    def test_resume_truncates_speculative_rows_after_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training_metrics.jsonl"
            path.write_text(
                '{"step": 2, "x": 1}\n{"step": 5, "x": 2}\n',
                encoding="utf-8",
            )
            _truncate_jsonl_after(path, 2)
            self.assertEqual(path.read_text(encoding="utf-8").splitlines(),
                             ['{"step": 2, "x": 1}'])

    def test_pending_checkpoint_allows_retry_of_active_identity(self):
        from utils.checkpointing.resume import find_latest_training_state

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "wandb_identity.json").write_text(
                '{"status":"active"}\n', encoding="utf-8"
            )
            state = root / "training-state-000005"
            state.mkdir()
            (state / ".model-pending").write_text("pending\n", encoding="utf-8")
            self.assertTrue(_has_pending_checkpoint_artifact(root))
            find_latest_training_state(root)
            self.assertFalse(state.exists())
            _validate_run_resume_progress(
                root, False, allow_pending_checkpoint_retry=True
            )

    def test_wandb_external_checkpoint_starts_a_new_run(self):
        fake_wandb = mock.Mock()
        fake_wandb.init.side_effect = _fake_wandb_run
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.dict(
                sys.modules, {"wandb": fake_wandb}
            ), mock.patch.dict(os.environ, {"WANDB_MODE": "online"}, clear=False):
                init_wandb(
                    _code_train_config(output_dir=directory, wandb=True),
                    resume=True,
                    resume_step=8,
                )
        self.assertNotIn("resume", fake_wandb.init.call_args.kwargs)

    def test_wandb_missing_identity_fails_for_progressed_directory(self):
        fake_wandb = mock.Mock()
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "manifest.json").write_text(
                "{}\n", encoding="utf-8"
            )
            with mock.patch.dict(sys.modules, {"wandb": fake_wandb}), mock.patch.dict(
                os.environ, {"WANDB_MODE": "offline"}, clear=False
            ):
                with self.assertRaisesRegex(ValueError, "identity is missing"):
                    init_wandb(_code_train_config(output_dir=directory, wandb=True))
            self.assertFalse(fake_wandb.init.called)

    def test_wandb_accepts_identity_environment_on_fresh_run(self):
        fake_wandb = mock.Mock()
        fake_wandb.init.side_effect = _fake_wandb_run
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            sys.modules, {"wandb": fake_wandb}
        ), mock.patch.dict(
            os.environ,
            {
                "WANDB_MODE": "offline",
                "WANDB_PROJECT": "env-project",
                "WANDB_NAME": "env-name",
                "WANDB_RUN_GROUP": "env-group",
                "WANDB_RUN_ID": "env123",
            },
            clear=False,
        ):
            init_wandb(
                _code_train_config(
                    output_dir=directory,
                    wandb=True,
                    wandb_run_name="config-name",
                    wandb_group="config-group",
                )
            )
            identity = json.loads(
                _wandb_identity_path(directory).read_text(encoding="utf-8")
            )
        self.assertEqual(identity["project"], "env-project")
        self.assertEqual(identity["name"], "env-name")
        self.assertEqual(identity["group"], "env-group")
        self.assertEqual(identity["run_id"], "env123")

    def test_wandb_rejects_stale_identity_environment(self):
        fake_wandb = mock.Mock()
        fake_wandb.init.side_effect = _fake_wandb_run
        with tempfile.TemporaryDirectory() as directory:
            config = _code_train_config(
                output_dir=directory,
                wandb=True,
                wandb_run_name="stable-name",
                wandb_group=None,
            )
            with mock.patch.dict(
                sys.modules, {"wandb": fake_wandb}
            ), mock.patch.dict(os.environ, {"WANDB_MODE": "offline"}, clear=False):
                init_wandb(config)
                with mock.patch.dict(
                    os.environ,
                    {"WANDB_MODE": "offline", "WANDB_RUN_ID": "wrong-id"},
                    clear=False,
                ):
                    with self.assertRaisesRegex(ValueError, "WANDB_RUN_ID conflicts"):
                        init_wandb(config, resume=True)
        self.assertEqual(fake_wandb.init.call_count, 1)

    def test_wandb_initial_environment_identity_is_inherited_on_restart(self):
        fake_wandb = mock.Mock()
        fake_wandb.init.side_effect = _fake_wandb_run
        with tempfile.TemporaryDirectory() as directory:
            config = _code_train_config(
                output_dir=directory,
                wandb=True,
                wandb_run_name="config-name",
                wandb_group="config-group",
            )
            with mock.patch.dict(sys.modules, {"wandb": fake_wandb}):
                with mock.patch.dict(
                    os.environ,
                    {
                        "WANDB_MODE": "offline",
                        "WANDB_PROJECT": "env-project",
                        "WANDB_NAME": "env-name",
                        "WANDB_RUN_GROUP": "env-group",
                        "WANDB_ENTITY": "env-entity",
                        "WANDB_RUN_ID": "env123",
                    },
                    clear=True,
                ):
                    init_wandb(config)
                # A launcher need not repeat the initial WANDB_* overrides.
                # The sidecar remains the source of truth for continuation.
                with mock.patch.dict(
                    os.environ, {"WANDB_MODE": "offline"}, clear=True
                ):
                    init_wandb(config, resume=True, resume_step=4)

        first = fake_wandb.init.call_args_list[0].kwargs
        second = fake_wandb.init.call_args_list[1].kwargs
        for key in ("project", "name", "group", "entity", "id"):
            self.assertEqual(second[key], first[key])

    def test_disabled_wandb_does_not_create_run_identity(self):
        fake_wandb = mock.Mock()
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            sys.modules, {"wandb": fake_wandb}
        ), mock.patch.dict(os.environ, {"WANDB_MODE": "disabled"}, clear=False):
            self.assertIsNone(
                init_wandb(_code_train_config(output_dir=directory, wandb=True))
            )
            self.assertFalse(_wandb_identity_path(directory).exists())
            self.assertFalse(fake_wandb.init.called)

    def test_disabled_environment_overrides_configured_mode(self):
        config = _code_train_config(wandb=True, wandb_mode="online")
        with mock.patch.dict(os.environ, {"WANDB_MODE": "disabled"}, clear=False):
            self.assertEqual(_requested_wandb_mode(config), "disabled")

    def test_external_legacy_wandb_directory_is_not_migrated(self):
        fake_wandb = mock.Mock()
        fake_wandb.init.side_effect = _fake_wandb_run
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as external:
            legacy_dir = Path(external) / "wandb"
            legacy_dir.mkdir()
            (legacy_dir / "run_id").write_text("foreign123\n", encoding="utf-8")
            with mock.patch.dict(sys.modules, {"wandb": fake_wandb}), mock.patch.dict(
                os.environ, {"WANDB_MODE": "offline"}, clear=False
            ):
                init_wandb(
                    _code_train_config(
                        output_dir=directory,
                        wandb=True,
                        wandb_dir=str(legacy_dir),
                    )
                )
            identity = json.loads(
                _wandb_identity_path(directory).read_text(encoding="utf-8")
            )
        self.assertNotEqual(identity["run_id"], "foreign123")

    def test_wandb_rejects_run_without_core_metadata(self):
        fake_wandb = mock.Mock()
        fake_wandb.init.return_value = object()
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.dict(sys.modules, {"wandb": fake_wandb}), mock.patch.dict(
                os.environ, {"WANDB_MODE": "offline"}, clear=False
            ):
                with self.assertRaisesRegex(RuntimeError, "metadata"):
                    init_wandb(_code_train_config(output_dir=directory, wandb=True))

    def test_active_wandb_identity_blocks_step_zero_restart(self):
        fake_wandb = mock.Mock()
        fake_wandb.init.side_effect = _fake_wandb_run
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.dict(sys.modules, {"wandb": fake_wandb}), mock.patch.dict(
                os.environ, {"WANDB_MODE": "offline"}, clear=False
            ):
                init_wandb(_code_train_config(output_dir=directory, wandb=True))
            with self.assertRaisesRegex(ValueError, "active W&B identity"):
                _validate_run_resume_progress(directory, has_checkpoint=False)

    def test_pending_wandb_identity_allows_metadata_retry(self):
        fake_wandb = mock.Mock()
        fake_wandb.init.side_effect = _fake_wandb_run
        with tempfile.TemporaryDirectory() as directory:
            config = _code_train_config(output_dir=directory, wandb=True)
            with mock.patch.dict(sys.modules, {"wandb": fake_wandb}):
                init_wandb(config, activate=False)
            identity = json.loads(
                _wandb_identity_path(directory).read_text(encoding="utf-8")
            )
            self.assertEqual(identity["status"], "pending")
            (Path(directory) / "manifest.json").write_text(
                "{}\n", encoding="utf-8"
            )
            _validate_run_resume_progress(directory, has_checkpoint=False)

    def test_wandb_refuses_restarting_without_checkpoint_after_logging(self):
        fake_wandb = mock.Mock()
        fake_wandb.init.side_effect = _fake_wandb_run
        with tempfile.TemporaryDirectory() as directory:
            config = _code_train_config(output_dir=directory, wandb=True)
            with mock.patch.dict(sys.modules, {"wandb": fake_wandb}):
                init_wandb(config)
                (Path(directory) / "training_metrics.jsonl").write_text(
                    '{"step": 1}\n', encoding="utf-8"
                )
                with self.assertRaisesRegex(ValueError, "no resumable checkpoint"):
                    _validate_run_resume_progress(directory, has_checkpoint=False)

    def test_resume_progress_rejects_manifest_without_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "manifest.json").write_text(
                "{}\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(
                ValueError, "manifest but no resumable checkpoint"
            ):
                _validate_run_resume_progress(directory, has_checkpoint=False)

    def test_wandb_rejects_training_protocol_change(self):
        fake_wandb = mock.Mock()
        fake_wandb.init.side_effect = _fake_wandb_run
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.dict(sys.modules, {"wandb": fake_wandb}):
                init_wandb(_code_train_config(output_dir=directory, wandb=True))
                with self.assertRaisesRegex(
                    ValueError, "initial config mismatch for learning_rate"
                ):
                    init_wandb(
                        _code_train_config(
                            output_dir=directory, wandb=True, learning_rate=1e-5
                        )
                    )

    def test_wandb_can_start_after_an_explicitly_disabled_run(self):
        fake_wandb = mock.Mock()
        fake_wandb.init.side_effect = _fake_wandb_run
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.json").write_text(
                json.dumps({"effective_config": {"wandb": False}}),
                encoding="utf-8",
            )
            with mock.patch.dict(sys.modules, {"wandb": fake_wandb}):
                init_wandb(_code_train_config(output_dir=directory, wandb=True))
            self.assertTrue(_wandb_identity_path(directory).is_file())

    def test_wandb_identity_rejects_project_change(self):
        fake_wandb = mock.Mock()
        fake_wandb.init.side_effect = _fake_wandb_run
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.dict(sys.modules, {"wandb": fake_wandb}):
                init_wandb(_code_train_config(output_dir=directory, wandb=True))
                with self.assertRaisesRegex(ValueError, "identity mismatch for project"):
                    init_wandb(
                        _code_train_config(
                            output_dir=directory,
                            wandb=True,
                            wandb_project="another-project",
                        )
                    )

    def test_training_metrics_are_append_only_jsonl(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training_metrics.jsonl"
            _append_training_metrics(path, 16, {"benchmark/mbpp": 0.25})
            _append_training_metrics(path, 32, {"train/reward": 1.0})
            rows = [json.loads(line) for line in path.read_text().splitlines()]

        self.assertEqual(rows, [
            {"benchmark/mbpp": 0.25, "step": 16},
            {"step": 32, "train/reward": 1.0},
        ])

    def test_online_eval_inherits_selected_training_fields(self):
        config = _code_train_config(
            gen_length=128,
            gen_steps=256,
            seed=77,
            mtp_block_size=3,
            mtp_confidence="max_probability",
            mtp_threshold=0.6,
        )
        raw_eval_config = dict(config.eval_config)
        config.validate()

        self.assertEqual(config.eval_config, raw_eval_config)
        resolved = resolve_online_eval_config(config)
        self.assertEqual(resolved["gen_length"], 128)
        self.assertEqual(resolved["seed"], 77)
        self.assertEqual(resolved["mtp_block_size"], 3)
        self.assertEqual(resolved["mtp_confidence"], "max_probability")
        self.assertEqual(resolved["mtp_threshold"], 0.6)
        self.assertEqual(resolved["generation_method"], "diffusion")
        self.assertEqual(resolved["batch_size"], 1)
        self.assertEqual(resolved["output_dir"], "/runs/train/benchmark")

    def test_effective_eval_constraints_fail_before_training(self):
        config = _code_train_config(
            gen_length=250,
            gen_steps=500,
            eval_config={
                "tasks": ["mbpp"],
                "mbpp_data_file": "/datasets/mbpp.jsonl",
                "mbpp_test_file": "/datasets/mbpp_test.jsonl",
                "gen_length": 250,
            },
        )
        with self.assertRaisesRegex(ValueError, "divisible by block_length"):
            config.validate()


if __name__ == "__main__":
    unittest.main()
