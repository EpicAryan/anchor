import os

from anchor.walker import (MAX_FILE_BYTES, is_indexable, is_secret_file,
                           iter_files, passes_static_gates)


def make_tree(tmp_path):
    root = tmp_path / "project"
    (root / "src").mkdir(parents=True)
    (root / "node_modules" / "lib").mkdir(parents=True)
    (root / ".git").mkdir()
    (root / "docs").mkdir()
    (root / "notes.md").write_text("top note")
    (root / "src" / "app.py").write_text("print('hi')")
    (root / "docs" / "guide.txt").write_text("nested note")
    (root / "node_modules" / "lib" / "dep.js").write_text("junk")
    (root / ".git" / "config.ini").write_text("git internals")
    (root / ".env").write_text("API_KEY=supersecret")
    (root / "server.pem").write_text("---BEGIN---")
    (root / "readme.docx").write_text("unsupported type")
    return root


def test_iter_files_recurses_and_excludes(tmp_path):
    root = make_tree(tmp_path)
    names = {p.name for p in iter_files(root)}
    assert names == {"notes.md", "app.py", "guide.txt"}


def test_iter_files_is_deterministic(tmp_path):
    root = make_tree(tmp_path)
    assert list(iter_files(root)) == list(iter_files(root))


def test_secret_files_blocked(tmp_path):
    root = make_tree(tmp_path)
    assert is_secret_file(root / ".env")
    assert is_secret_file(root / ".env.production")
    assert is_secret_file(root / "server.pem")
    assert is_secret_file(root / "id_rsa")
    assert is_secret_file(root / "deploy.key")
    assert is_secret_file(root / "credentials.json")
    assert is_secret_file(root / "SECRETS.yaml")      # case-insensitive
    assert not is_secret_file(root / "notes.md")
    assert not is_secret_file(root / "app.py")


def test_symlink_not_indexable(tmp_path):
    root = make_tree(tmp_path)
    outside = tmp_path / "outside.md"
    outside.write_text("outside content")
    link = root / "link.md"
    link.symlink_to(outside)
    assert not is_indexable(link, root)


def test_path_escaping_root_not_indexable(tmp_path):
    root = make_tree(tmp_path)
    outside = tmp_path / "elsewhere.md"
    outside.write_text("x")
    assert not is_indexable(root / ".." / "elsewhere.md", root)


def test_oversized_file_not_indexable(tmp_path):
    root = make_tree(tmp_path)
    big = root / "big.md"
    big.write_text("x")
    os.truncate(big, MAX_FILE_BYTES + 1)     # sparse file, no real disk usage
    assert not is_indexable(big, root)


def test_pdf_gets_larger_cap(tmp_path):
    root = make_tree(tmp_path)
    pdf = root / "scan.pdf"
    pdf.write_bytes(b"%PDF-")
    os.truncate(pdf, MAX_FILE_BYTES + 1)     # over text cap, under PDF cap
    assert is_indexable(pdf, root)


def test_static_gates_work_for_deleted_paths(tmp_path):
    root = make_tree(tmp_path)
    assert passes_static_gates(root / "gone.md", root)          # never existed
    assert not passes_static_gates(root / "gone.env", root)
    assert not passes_static_gates(root / "node_modules" / "x.js", root)
    assert not passes_static_gates(tmp_path / "outside.md", root)


def test_excluded_dir_file_not_indexable_even_directly(tmp_path):
    root = make_tree(tmp_path)
    assert not is_indexable(root / "node_modules" / "lib" / "dep.js", root)
    assert not is_indexable(root / ".git" / "config.ini", root)
