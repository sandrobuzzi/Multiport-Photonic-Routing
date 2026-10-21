"""Scientific invariants and data-import contracts; no measurement fixtures."""

from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "workflow"))
sys.path.insert(0, str(ROOT / "scripts"))
import waveforms
import model_v1
import download_data


class WaveformTests(unittest.TestCase):
    def setUp(self):
        self.states = {
            "red": {"awg_v": np.array([1.0, -2.0, 0.5])},
            "green": {"awg_v": np.array([-0.5, 1.0, 2.0])},
            "blue": {"awg_v": np.array([0.2, -0.5, -1.0])},
        }

    def test_strobe_is_dc_balanced_and_within_range(self):
        values, states, names = waveforms.build_strobe_waveform(
            self.states, n_idle_slots=2
        )
        self.assertEqual(names, ["RED", "BLUE", "GREEN", "IDLE", "IDLE"])
        np.testing.assert_allclose(values.mean(axis=1), 0, atol=1e-14)
        self.assertLessEqual(np.abs(values).max(), 1)

    def test_fractional_samples_per_cycle_rejected(self):
        with self.assertRaises(ValueError):
            waveforms.build_strobe_waveform(
                self.states, n_idle_slots=2, cycle_freq_hz=2.4e9 / 240.25
            )

    def test_invalid_waveform_settings_are_rejected(self):
        for idle in (0, -1, 1.5):
            with self.subTest(idle=idle), self.assertRaises(ValueError):
                waveforms.build_strobe_waveform(self.states, n_idle_slots=idle)
        with self.assertRaises(ValueError):
            waveforms.build_cube_trajectory(f_hi_hz=waveforms.AWG_SR / 2)
        with self.assertRaises(ValueError):
            waveforms.build_cube_trajectory(n_samples=100)

    def test_uniform_remap_preserves_rank_and_uniform_marginals(self):
        signals = np.array([[3.0, -2.0, 0.2, 1.0], [-4.0, 0.0, 2.0, -1.0]])
        remapped, _ = waveforms.normalize_uniform_remap(signals)
        for original, result in zip(signals, remapped):
            np.testing.assert_array_equal(np.argsort(original), np.argsort(result))
            np.testing.assert_allclose(np.sort(result), np.linspace(-1, 1, 4))

    def test_balanced_mmi_is_unitary_and_reciprocal_round_trip(self):
        mmi = model_v1.make_mmi(np.zeros(3), np.ones(2))
        np.testing.assert_allclose(mmi.conj().T @ mmi, np.eye(3), atol=1e-14)
        reflectors = np.diag(np.exp(1j * np.array([0.2, -1.1, 2.7])))
        round_trip = mmi.T @ reflectors @ mmi
        np.testing.assert_allclose(round_trip, round_trip.T, atol=1e-14)
        np.testing.assert_allclose(
            np.sum(np.abs(round_trip) ** 2, axis=0), 1, atol=1e-14
        )


class DownloadTests(unittest.TestCase):
    def test_only_data_files_are_extracted(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            archive = base / "archive.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("release/data/fig4/example.json", "{}")
                bundle.writestr("release/data/fig4/script.py", "raise RuntimeError()")
                bundle.writestr("release/viewer/data/arrays.json", "{}")
            self.assertEqual(download_data.extract_data(archive, base / "data"), 1)
            self.assertTrue((base / "data/fig4/example.json").is_file())
            self.assertFalse((base / "data/fig4/script.py").exists())

    def test_archive_cannot_escape_data_directory(self):
        with self.assertRaises(ValueError):
            download_data.archive_data_path("release/data/../../example.json")


if __name__ == "__main__":
    unittest.main()
