import io

import zerograd._ui as ui


def test_quiet_env_silences_everything(monkeypatch):
    monkeypatch.setenv("ZEROGRAD_QUIET", "1")
    out = io.StringIO()
    c = ui.Console(True, stream=out)
    c.say("hello")
    with c.spinner("work", done="done"):
        pass
    assert out.getvalue() == ""


def test_plain_output_off_a_terminal(monkeypatch):
    monkeypatch.delenv("ZEROGRAD_QUIET", raising=False)
    out = io.StringIO()
    c = ui.Console(True, stream=out)
    assert not c.live and not c.color
    with c.spinner("Scoring", done="Scored"):
        pass
    text = out.getvalue()
    assert "… Scoring" in text and "✓ Scored" in text and "\r" not in text and "\033[" not in text


def test_no_color(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    out = io.StringIO()
    out.isatty = lambda: True
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    c = ui.Console(True, stream=out)
    assert c.live and not c.color


def test_greets_once_with_a_phrase(monkeypatch):
    monkeypatch.delenv("ZEROGRAD_QUIET", raising=False)
    monkeypatch.setattr(ui, "_greeted", False)
    out = io.StringIO()
    c = ui.Console(True, stream=out)
    c.greet("local", "2 models ready")
    c.greet("local", "2 models ready")
    lines = out.getvalue().splitlines()
    assert len(lines) == 2 and "ZeroGrad" in lines[0] and lines[1].strip() in ui.PHRASES["init"]


def test_enough_phrases_for_every_moment():
    assert sum(len(v) for v in ui.PHRASES.values()) >= 15
    assert "Look mama, no gradients!!" in ui.PHRASES["init"]
    assert {"init", "download", "championship", "fit", "predict", "champion"} <= set(ui.PHRASES)


def test_podium_orders_and_marks_failures():
    out = io.StringIO()
    c = ui.Console(True, stream=out)
    c.podium([("A", 0.9, None), ("B", 0.8, None), ("C", None, "boom")], "ROC AUC", "Built with X", baseline=0.5)
    text = out.getvalue()
    assert text.index("🥇") < text.index("🥈") and "did not finish" in text and "Champion: A" in text
