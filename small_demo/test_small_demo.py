from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
import subprocess
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_DIR = ROOT / "monteCarloPassSearch/monteCarloPassSearch/initParamVar"
SMART_DIR = ROOT / "monteCarloPassSearch/trajModel_smart"
for value in (str(RUNTIME_DIR), str(SMART_DIR)):
    if value not in sys.path:
        sys.path.insert(0, value)

from sim_viz_runtime import smart_predict_positions  # noqa: E402


class FakeTokenizer:
    def encode_player(self, values):
        return np.zeros(values.shape[:-1], dtype=np.int64)

    def encode_ball(self, values):
        return np.zeros(values.shape[:-1], dtype=np.int64)

    def decode_player(self, token_ids):
        token_ids = np.asarray(token_ids)
        result = np.zeros(token_ids.shape + (10,), dtype=np.float32)
        result[token_ids == 1] = 0.05
        return result


class CandidateAwareModel:
    def __call__(self, player_tokens, ball_tokens, player_pos, ball_pos, entity_types, obs_mask, perm=None):
        batch, steps, players = player_tokens.shape
        logits = torch.zeros((batch, steps, players, 2), device=player_tokens.device)
        choose_one = ball_pos[:, -1, 0] > 0
        logits[..., 0] = 5.0
        logits[choose_one, ..., 0] = -5.0
        logits[choose_one, ..., 1] = 5.0
        ball_logits = torch.zeros((batch, steps, 2), device=player_tokens.device)
        return logits, ball_logits


class CounterfactualLeakageTest(unittest.TestCase):
    def make_scene(self):
        features = np.zeros((140, 23, 6), dtype=np.float32)
        entity_type = np.array([0] * 11 + [1] * 11 + [2], dtype=np.int64)
        mask = np.ones((140, 23), dtype=bool)
        for player in range(22):
            features[:, player, 0] = player - 11
            features[:, player, 1] = player % 5
        return features, entity_type, mask

    def rollout(self, features, candidate):
        return smart_predict_positions(
            (CandidateAwareModel(), FakeTokenizer(), {"history": 8, "rollout": 4}),
            features, self.entity_type, self.mask,
            ctx_start=0, kfl=80, sim_len=20, device=torch.device("cpu"),
            candidate_ball_pos=candidate, top_k=1, seed=7,
        )

    def test_future_recording_is_not_read(self):
        features, self.entity_type, self.mask = self.make_scene()
        candidate = np.zeros((21, 3), dtype=np.float32)
        first = self.rollout(features, candidate)
        features[81:, :22, :2] = 9999
        features[81:, 22, :3] = -9999
        second = self.rollout(features, candidate)
        np.testing.assert_allclose(first, second)

    def test_candidate_ball_changes_rollout(self):
        features, self.entity_type, self.mask = self.make_scene()
        left = np.zeros((21, 3), dtype=np.float32)
        left[:, 0] = -10
        right = np.zeros((21, 3), dtype=np.float32)
        right[:, 0] = 10
        self.assertFalse(np.allclose(self.rollout(features, left), self.rollout(features, right)))


class CapDataTest(unittest.TestCase):
    def test_pv_cap_keeps_positives(self):
        spec = importlib.util.spec_from_file_location("cap_datasets", ROOT / "small_demo/cap_datasets.py")
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            labels = np.zeros((20, 4), dtype=np.float32)
            labels[:3, 2] = 1
            arrays = {
                "features": np.zeros((20, 2, 23, 6), np.float32),
                "mask": np.ones((20, 2, 23), np.uint8),
                "labels": labels,
                "entity_type": np.arange(23),
            }
            chunk = directory / "chunk.npz"
            np.savez_compressed(chunk, **arrays)
            (directory / "manifest.json").write_text(json.dumps([{"path": str(chunk), "n_windows": 20}]))
            report = module.cap_pv(directory, cap=8, seed=1)
            self.assertEqual(report["after"], 8)
            self.assertEqual(report["positives"], 3)


class ChronologicalSplitTest(unittest.TestCase):
    def test_boundary_windows_are_dropped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            output = root / "output"
            (source / "train").mkdir(parents=True)
            rows = []
            # Short, non-overlapping clips plus two clips crossing the cut points.
            intervals = [(0, 9), (10, 19), (20, 29), (25, 45), (46, 59), (55, 75), (76, 89), (90, 99)]
            for idx, (start, end) in enumerate(intervals):
                path = source / "train" / f"clip_{idx}.npz"
                length = end - start + 1
                np.savez_compressed(
                    path,
                    frames=np.arange(start, end + 1),
                    periods=np.ones(length),
                    times=np.arange(start, end + 1, dtype=np.float32),
                )
                rows.append({"clip_path": str(path), "length": 400, "clip_index": idx})
            with (source / "train/manifest.jsonl").open("w") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")
            subprocess.run([
                sys.executable, str(ROOT / "small_demo/split_one_match.py"),
                "--source-root", str(source), "--out-root", str(output),
                "--ratios", "0.4,0.3,0.3", "--train-smart-windows", "9999",
                "--eval-smart-windows", "9999",
            ], check=True, capture_output=True, text=True)
            report = json.loads((output / "split_report.json").read_text())
            self.assertEqual(report["dropped_boundary_clips"], 2)
            for split in ("train", "val", "test"):
                self.assertGreater(report["splits"][split]["clips"], 0)


if __name__ == "__main__":
    unittest.main()
