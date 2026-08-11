from photosearch.scanner import scan_for_changes


def _touch(p, content=b"x"):
    p.write_bytes(content)


def test_scan_finds_new_and_changed_skips_unchanged(db, tmp_path):
    share = tmp_path / "share"
    share.mkdir()
    a = share / "a.jpg"
    _touch(a)
    b = share / "b.png"
    _touch(b)
    note = share / "notes.txt"
    _touch(note)  # ignored ext

    first = scan_for_changes(db, [share])
    assert set(first) == {a, b}

    # record a as ingested
    st = a.stat()
    db.execute(
        "INSERT INTO photos(id, path, content_hash, mtime, size) VALUES (?,?,?,?,?)",
        ("id-a", str(a), "h", st.st_mtime, st.st_size),
    )
    db.commit()

    second = scan_for_changes(db, [share])
    assert a not in second and b in second

    # modify a -> reappears
    _touch(a, b"xxxx")
    third = scan_for_changes(db, [share])
    assert a in third
