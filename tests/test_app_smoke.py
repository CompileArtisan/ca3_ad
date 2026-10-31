from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def _slider(at, label_start):
    return next(s for s in at.sidebar.slider if s.label.startswith(label_start))


def test_app_runs_without_exception():
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception, at.exception


def test_disclaimer_banner_present():
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert any("computational abstraction" in i.value for i in at.info)


def test_every_heavy_suite_runs_from_the_ui():
    at = AppTest.from_file(APP, default_timeout=300).run()
    _slider(at, "Trials").set_value(5).run()
    _slider(at, "Synaptic weakening").set_value(0.3).run()
    _slider(at, "Neuronal loss").set_value(0.2).run()
    for label in ("Run damage sweeps (A+B+C)", "Run combined-damage grid (D)",
                  "Run new-learning experiments (E)", "Run memory-load sweep (F)"):
        btn = next(b for b in at.button if b.label == label)
        btn.click().run()
        assert not at.exception, (label, at.exception)
    for key in ("res_sweeps", "res_combined", "res_learning", "res_load"):
        assert key in at.session_state
