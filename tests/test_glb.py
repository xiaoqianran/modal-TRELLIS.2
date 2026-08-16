from modal_trellis2.core.glb import GlbError, colored_cube_glb, is_glb, validate_glb


def test_colored_cube_is_valid_glb() -> None:
    glb = colored_cube_glb((0.77, 0.42, 0.23))
    assert is_glb(glb)
    validate_glb(glb)
    assert glb[:4] == b"glTF"
    assert len(glb) > 400


def test_validate_rejects_garbage() -> None:
    try:
        validate_glb(b"not a mesh")
    except GlbError:
        return
    raise AssertionError("expected GlbError")
