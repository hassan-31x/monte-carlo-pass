#!/usr/bin/env python3
"""Colab-oriented end-to-end runner for a small MCPS experiment."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "monteCarloPassSearch"
HERE = Path(__file__).resolve().parent


class PipelineStageError(RuntimeError):
    """Failure with enough context to diagnose it from a Colab cell."""

    def __init__(self, name: str, command: list[str], code: int, log_path: Path) -> None:
        self.name = name
        self.command = command
        self.code = code
        self.log_path = log_path
        super().__init__(f"Pipeline stage {name!r} failed with exit code {code}")


class Pipeline:
    def __init__(self, config_path: Path) -> None:
        self.config_path = config_path.resolve()
        self.config = json.loads(self.config_path.read_text(encoding="utf-8"))
        required_research_file = (
            RESEARCH / "ballPlayerTrajModel/publicData/kloppy_to_preprocessed.py"
        )
        if not required_research_file.is_file():
            raise FileNotFoundError(
                "The MCPS research sources are missing from this checkout. "
                f"Expected: {required_research_file}. If this is Colab and the repository "
                "directory already existed, run `git pull --ff-only` in the repository, "
                "or delete the stale Colab checkout and rerun the clone cell."
            )
        work = Path(self.config["work_dir"])
        self.work = work if work.is_absolute() else ROOT / work
        self.logs = self.work / "logs"
        self.logs.mkdir(parents=True, exist_ok=True)
        self.compute_path = self.work / "compute_report.json"
        self.compute = self._read_json(self.compute_path, {"stages": []})

    @staticmethod
    def _read_json(path: Path, default: object) -> object:
        return json.loads(path.read_text()) if path.exists() else default

    def _gpu_snapshot(self) -> dict:
        command = [
            "nvidia-smi", "--query-gpu=name,memory.total,memory.used",
            "--format=csv,noheader,nounits",
        ]
        try:
            output = subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL).strip()
        except (FileNotFoundError, subprocess.CalledProcessError):
            return {"available": False}
        return {"available": True, "devices": output.splitlines()}

    @staticmethod
    def _gpu_used_memory() -> list[int]:
        try:
            output = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                text=True, stderr=subprocess.DEVNULL,
            )
            return [int(value.strip()) for value in output.splitlines() if value.strip()]
        except (FileNotFoundError, subprocess.CalledProcessError, ValueError):
            return []

    def run(self, name: str, command: list[str], cwd: Path | None = None) -> None:
        log_path = self.logs / f"{name}.log"
        started = time.time()
        gpu_before = self._gpu_snapshot()
        peak_gpu = self._gpu_used_memory()
        stop_polling = threading.Event()

        def poll_gpu() -> None:
            nonlocal peak_gpu
            while not stop_polling.wait(1.0):
                values = self._gpu_used_memory()
                if len(values) > len(peak_gpu):
                    peak_gpu.extend([0] * (len(values) - len(peak_gpu)))
                peak_gpu = [max(old, new) for old, new in zip(peak_gpu, values)]

        poller = threading.Thread(target=poll_gpu, daemon=True)
        poller.start()
        print(f"\n[{name}] {' '.join(command)}", flush=True)
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                cwd=str(cwd or ROOT),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="")
                log.write(line)
            code = process.wait()
        stop_polling.set()
        poller.join(timeout=2.0)
        elapsed = time.time() - started
        record = {
            "name": name,
            "command": command,
            "seconds": elapsed,
            "return_code": code,
            "gpu_before": gpu_before,
            "gpu_after": self._gpu_snapshot(),
            "peak_gpu_memory_mib": peak_gpu,
            "log": str(log_path),
        }
        self.compute["stages"] = [row for row in self.compute["stages"] if row["name"] != name]
        self.compute["stages"].append(record)
        self.compute_path.write_text(json.dumps(self.compute, indent=2), encoding="utf-8")
        if code:
            raise PipelineStageError(name, command, code, log_path)

    def prepare(self) -> None:
        raw = self.work / "tracking_raw_split"
        data = self.work / "tracking"
        ptt = self.work / "ptt_data"
        bat = self.work / "bat_data"
        pv = self.work / "pv_data"
        vocabs = self.work / "vocabs"
        shots = self.work / "sportec_shots.json"
        cap = self.config["caps"]
        ratios = ",".join(str(value) for value in self.config["split_ratios"])

        self.run("01_tracking_preprocess", [
            sys.executable,
            str(RESEARCH / "ballPlayerTrajModel/publicData/kloppy_to_preprocessed.py"),
            "--provider", "sportec",
            "--match-ids", self.config["match_id"],
            "--out-dir", str(raw),
            "--history-frames", str(self.config["history_frames"]),
            "--future-frames", str(self.config["future_frames"]),
            "--max-frames-per-match", str(self.config["max_frames"]),
            "--ball-mode", "xyz",
            "--val-ratio", "0",
            "--test-ratio", "0",
            "--seed", str(self.config["seed"]),
        ])
        self.run("02_chronological_split", [
            sys.executable, str(HERE / "split_one_match.py"),
            "--source-root", str(raw), "--out-root", str(data),
            "--ratios", ratios,
            "--train-smart-windows", str(cap["smart_train_windows"]),
            "--eval-smart-windows", str(cap["smart_eval_windows"]),
            "--seed", str(self.config["seed"]),
        ])
        self.run("03_build_vocab", [
            sys.executable, str(RESEARCH / "trajModel_smart/build_vocab.py"),
            "--preprocessed-dir", str(data), "--output-dir", str(vocabs),
            "--seed", str(self.config["seed"]),
        ])
        self.run("04_extract_shots", [
            sys.executable, str(HERE / "extract_sportec_shots.py"),
            "--match-id", self.config["match_id"], "--output", str(shots),
        ])
        self.run("05_preprocess_touch", [
            sys.executable, str(RESEARCH / "playerToTouch/preprocess_sportec.py"),
            "--sportec-dir", str(data), "--out-dir", str(ptt),
            "--mode", "survival", "--seed", str(self.config["seed"]),
        ])
        self.run("06_preprocess_bat", [
            sys.executable, str(RESEARCH / "ballAtTouch/initParamVar/preprocess_sportec.py"),
            "--sportec-dir", str(data), "--out-dir", str(bat), "--all-touches",
        ])
        self.run("07_preprocess_pv", [
            sys.executable, str(RESEARCH / "possessionValue/preprocess_pv.py"),
            "--preprocessed-dir", str(data), "--sportec-shots", str(shots),
            "--out-dir", str(pv), "--window-size", "64", "--stride", "32",
            "--horizon-frames", "250",
        ])
        self.run("08_cap_datasets", [
            sys.executable, str(HERE / "cap_datasets.py"),
            "--ptt-root", str(ptt), "--bat-root", str(bat), "--pv-root", str(pv),
            "--touch-train", str(cap["touch_train"]), "--touch-eval", str(cap["touch_eval"]),
            "--bat-train", str(cap["bat_train"]), "--bat-eval", str(cap["bat_eval"]),
            "--pv-train", str(cap["pv_train"]), "--pv-eval", str(cap["pv_eval"]),
            "--seed", str(self.config["seed"]), "--report", str(self.work / "dataset_report.json"),
        ])

    def train(self) -> None:
        training = self.config["training"]
        checkpoints = self.work / "checkpoints"
        seed = str(self.config["seed"])
        epochs = str(training["epochs"])
        workers = str(training["num_workers"])
        self.run("09_train_smart", [
            sys.executable, str(RESEARCH / "trajModel_smart/train.py"),
            "--preprocessed-dir", str(self.work / "tracking"),
            "--vocab-dir", str(self.work / "vocabs"),
            "--save-dir", str(checkpoints), "--run-name", "smart",
            "--epochs", epochs, "--batch-size", str(training["smart_batch_size"]),
            "--num-workers", workers, "--warmup-steps", "50", "--seed", seed,
        ])
        self.run("10_train_touch", [
            sys.executable, str(RESEARCH / "playerToTouch/train.py"),
            "--preprocessed-dir", str(self.work / "ptt_data"),
            "--save-dir", str(checkpoints), "--run-name", "touch",
            "--model-type", "survival", "--epochs", epochs,
            "--batch-size", str(training["touch_batch_size"]),
            "--num-workers", workers, "--augment", "--seed", seed,
        ])
        self.run("11_train_bat", [
            sys.executable, str(RESEARCH / "ballAtTouch/initParamVar/train.py"),
            "--preprocessed-dir", str(self.work / "bat_data"),
            "--save-dir", str(checkpoints), "--run-name", "bat",
            "--model-type", "gaussian", "--epochs", epochs,
            "--batch-size", str(training["bat_batch_size"]),
            "--num-workers", workers, "--augment", "--seed", seed,
        ])
        self.run("12_train_pv", [
            sys.executable, str(RESEARCH / "possessionValue/train_pv.py"),
            "--data-dir", str(self.work / "pv_data"),
            "--checkpoint-dir", str(checkpoints / "pv"),
            "--epochs", epochs, "--batch-size", str(training["pv_batch_size"]),
            "--patience", "5", "--seed", seed,
        ])
        self.run("13_train_set_piece", [
            sys.executable, str(RESEARCH / "possessionValue/train_set_piece_pv.py"),
            "--out-dir", str(checkpoints / "set_piece"),
            "--sportec-shots", str(self.work / "sportec_shots.json"),
            "--no-statsbomb", "--seed", seed,
        ])

    def search(self) -> None:
        search = self.config["search"]
        output = self.work / "search"
        output.mkdir(parents=True, exist_ok=True)
        checkpoints = self.work / "checkpoints"
        self.run("14_mcps_search", [
            sys.executable,
            str(RESEARCH / "monteCarloPassSearch/initParamVar/pass_mc_runner.py"),
            "--manifest", str(self.work / "tracking/test/manifest.jsonl"),
            "--output-csv", str(output / "variants.csv"),
            "--resume",
            "--max-clips", str(search["max_clips"]),
            "--variants-local", str(search["local_variants"]),
            "--variants-global", str(search["global_variants"]),
            "--context-len", str(search["context_len"]),
            "--rollout-len", str(search["rollout_len"]),
            "--smart-checkpoint", str(checkpoints / "smart/best.pt"),
            "--vocab-dir", str(self.work / "vocabs"),
            "--touch-checkpoint", str(checkpoints / "touch/best.pt"),
            "--bat-checkpoint", str(checkpoints / "bat/best.pt"),
            "--pv-checkpoint", str(checkpoints / "pv/best.pt"),
            "--set-piece-pv-model", str(checkpoints / "set_piece/model.json"),
            "--gpus", "0", "--workers-per-gpu", "1",
            "--no-event-passer-labels", "--no-observed-use-gt-players",
            "--seed", str(self.config["seed"]),
        ])
        self.run("15_rank", [
            sys.executable,
            str(RESEARCH / "monteCarloPassSearch/initParamVar/rank_passers.py"),
            "--input-csv", str(output / "variants.csv"),
            "--output-csv", str(output / "rankings.csv"),
            "--output-csv-local", str(output / "rankings_local.csv"),
            "--output-csv-global", str(output / "rankings_global.csv"),
            "--clip-output-csv", str(output / "clip_percentiles.csv"),
            "--no-name-resolution",
        ])

    def report(self) -> None:
        figures = self.work / "figures"
        figures.mkdir(parents=True, exist_ok=True)
        self.run("16_best_of_20", [
            sys.executable, str(RESEARCH / "trajModel_smart/best_of_20_png.py"),
            "--checkpoint", str(self.work / "checkpoints/smart/best.pt"),
            "--preprocessed-dir", str(self.work / "tracking"),
            "--vocab-dir", str(self.work / "vocabs"),
            "--split", "test", "--window-idx", "0", "--num-samples", "20",
            "--num-gpus", "1", "--output", str(figures / "smart_best_of_20.png"),
            "--seed", str(self.config["seed"]),
        ])
        self.run("16_smart_animation", [
            sys.executable, str(RESEARCH / "trajModel_smart/viz.py"),
            "--checkpoint", str(self.work / "checkpoints/smart/best.pt"),
            "--preprocessed-dir", str(self.work / "tracking"),
            "--vocab-dir", str(self.work / "vocabs"),
            "--split", "test", "--window-idx", "0",
            "--output", str(figures / "smart_rollout.gif"), "--seed", str(self.config["seed"]),
        ])
        self.run("16_visualize", [
            sys.executable, str(HERE / "visualize_results.py"),
            "--work-dir", str(self.work),
        ])
        self.compute["total_seconds"] = sum(float(row["seconds"]) for row in self.compute["stages"])
        self.compute["configuration"] = self.config
        self.compute_path.write_text(json.dumps(self.compute, indent=2), encoding="utf-8")
        print(json.dumps({"work_dir": str(self.work), "compute_report": str(self.compute_path)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "train", "search", "report", "all"])
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    args = parser.parse_args()
    pipeline = Pipeline(args.config)
    try:
        if args.stage in {"prepare", "all"}:
            pipeline.prepare()
        if args.stage in {"train", "all"}:
            pipeline.train()
        if args.stage in {"search", "all"}:
            pipeline.search()
        if args.stage in {"report", "all"}:
            pipeline.report()
    except PipelineStageError as exc:
        try:
            lines = exc.log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            lines = []
        print("\n" + "=" * 78, file=sys.stderr)
        print(f"FAILED STAGE: {exc.name}", file=sys.stderr)
        print(f"EXIT CODE:    {exc.code}", file=sys.stderr)
        print(f"COMMAND:      {' '.join(exc.command)}", file=sys.stderr)
        print(f"FULL LOG:     {exc.log_path}", file=sys.stderr)
        if exc.code in {-9, 137}:
            print(
                "HINT: the process was killed, usually because the Colab runtime ran out of RAM.",
                file=sys.stderr,
            )
        if lines:
            print("\nLAST 80 LOG LINES:\n", file=sys.stderr)
            print("\n".join(lines[-80:]), file=sys.stderr)
        else:
            print("The stage produced no log output.", file=sys.stderr)
        print("=" * 78, file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
