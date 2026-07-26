import json

from paperlog.cli import main


def run(capsys, *argv):
    code = main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_build_writes_a_pdf_and_manifest(tmp_path, capsys):
    out = tmp_path / "j.pdf"
    code, stdout, _ = run(capsys, "build", "--pages", "4", "--out", str(out))
    assert code == 0
    assert out.exists()
    manifest = json.loads((tmp_path / "j.manifest.json").read_text())
    assert len(manifest["pages"]) == 4
    assert "wrote" in stdout


def test_dry_run_writes_nothing(tmp_path, capsys):
    out = tmp_path / "j.pdf"
    code, stdout, _ = run(capsys, "build", "--pages", "4", "--out", str(out), "--dry-run")
    assert code == 0
    assert not out.exists()
    assert "dry run" in stdout


def test_no_manifest_flag(tmp_path, capsys):
    out = tmp_path / "j.pdf"
    run(capsys, "build", "--pages", "2", "--out", str(out), "--no-manifest")
    assert out.exists()
    assert not (tmp_path / "j.manifest.json").exists()


def test_cli_overrides_beat_the_config_file(tmp_path, capsys):
    config = tmp_path / "j.yaml"
    config.write_text("pages: 100\nruling:\n  style: grid\n")
    out = tmp_path / "j.pdf"
    run(capsys, "build", "-c", str(config), "--pages", "6", "--out", str(out))
    manifest = json.loads((tmp_path / "j.manifest.json").read_text())
    assert len(manifest["pages"]) == 6
    assert manifest["page"]["ruling"] == "grid"


def test_preset_is_overridable(tmp_path, capsys):
    out = tmp_path / "j.pdf"
    run(capsys, "build", "--preset", "a5-dot", "--pages", "3", "--out", str(out))
    manifest = json.loads((tmp_path / "j.manifest.json").read_text())
    assert manifest["page"]["ruling"] == "dotted"
    assert len(manifest["pages"]) == 3


def test_no_qr_flag(tmp_path, capsys):
    out = tmp_path / "j.pdf"
    code, stdout, _ = run(capsys, "build", "--pages", "2", "--no-qr", "--out", str(out))
    assert code == 0
    assert "codes     none" in stdout
    manifest = json.loads((tmp_path / "j.manifest.json").read_text())
    assert manifest["qr"]["enabled"] is False


def test_notebook_id_is_reusable_for_reprints(tmp_path, capsys):
    for name in ("first", "second"):
        run(capsys, "build", "--pages", "2", "--notebook-id", "K7M2QX4A", "--out", str(tmp_path / f"{name}.pdf"))
    first = json.loads((tmp_path / "first.manifest.json").read_text())
    second = json.loads((tmp_path / "second.manifest.json").read_text())
    assert [p["token"] for p in first["pages"]] == [p["token"] for p in second["pages"]]


def test_bad_config_reports_an_error(tmp_path, capsys):
    config = tmp_path / "j.yaml"
    config.write_text("ruling:\n  style: squiggly\n")
    code, _, stderr = run(capsys, "build", "-c", str(config), "--out", str(tmp_path / "j.pdf"))
    assert code == 1
    assert "ruling.style" in stderr


def test_missing_config_file_reports_an_error(tmp_path, capsys):
    code, _, stderr = run(capsys, "build", "-c", str(tmp_path / "nope.yaml"))
    assert code == 1
    assert "no such file" in stderr


def test_warnings_go_to_stderr_but_still_build(tmp_path, capsys):
    out = tmp_path / "j.pdf"
    code, _, stderr = run(capsys, "build", "--pages", "2", "--qr-size", "5mm", "--out", str(out))
    assert code == 0
    assert out.exists()
    assert "below the" in stderr


def test_decode_command(capsys):
    code, stdout, _ = run(capsys, "decode", "PL1:K7M2QX4A:42:F:TR:DF6B")
    assert code == 0
    assert "top-right" in stdout
    assert "42 (front)" in stdout


def test_decode_command_rejects_a_bad_token(capsys):
    code, _, stderr = run(capsys, "decode", "PL1:K7M2QX4A:42:F:TR:0000")
    assert code == 1
    assert "checksum" in stderr


def test_decode_json_output(capsys):
    code, stdout, _ = run(capsys, "decode", "PL1:K7M2QX4A:42:F:TR:DF6B", "--json")
    assert code == 0
    assert json.loads(stdout)["page"] == 42


def test_presets_command_lists_sizes(capsys):
    code, stdout, _ = run(capsys, "presets")
    assert code == 0
    assert "a5-ruled" in stdout and "travelers" in stdout


def test_init_writes_a_usable_config(tmp_path, capsys):
    config = tmp_path / "j.yaml"
    run(capsys, "init", "-o", str(config))
    assert config.exists()
    # The starter config must actually build.
    code, _, _ = run(capsys, "build", "-c", str(config), "--pages", "2", "--out", str(tmp_path / "j.pdf"))
    assert code == 0


def test_init_refuses_to_clobber(tmp_path, capsys):
    config = tmp_path / "j.yaml"
    config.write_text("pages: 4\n")
    code, _, stderr = run(capsys, "init", "-o", str(config))
    assert code == 1
    assert "already exists" in stderr
    assert config.read_text() == "pages: 4\n"


def test_new_id_generates_distinct_ids(capsys):
    code, stdout, _ = run(capsys, "new-id", "-n", "5")
    ids = stdout.split()
    assert code == 0
    assert len(set(ids)) == 5
    assert all(len(value) == 8 for value in ids)
