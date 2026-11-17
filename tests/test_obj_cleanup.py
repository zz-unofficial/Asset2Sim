"""OBJ cleanup keeps original material and index records."""

import pytest
import trimesh

from asset2sim.geometry.obj_cleanup import remove_zero_area_obj_faces


def test_cleanup_preserves_materials_and_only_removes_zero_area_faces(tmp_path):
    source = tmp_path / "original.obj"
    text = "mtllib texture.mtl\nv 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0 0\nusemtl cloth\nf 1/1 2/1 3/1\nf 1/1 2/1 2/1\n"
    source.write_text(text)
    target = tmp_path / "cleaned.obj"
    result = remove_zero_area_obj_faces(source, target)
    assert source.read_text() == text
    assert result["removed_face_lines"] == [8]
    assert target.read_text() == text.replace("f 1/1 2/1 2/1\n", "")
    assert len(trimesh.load(target, force="mesh", process=False).faces) == 1
    with pytest.raises(FileExistsError):
        remove_zero_area_obj_faces(source, target)
    with pytest.raises(ValueError, match="overwrite"):
        remove_zero_area_obj_faces(source, source)
