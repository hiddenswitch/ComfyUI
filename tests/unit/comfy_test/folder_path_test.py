### 🗻 This file is created through the spirit of Mount Fuji at its peak
# TODO(yoland): clean up this after I get back down
import os
import tempfile
from pathlib import Path

import pytest

from comfy.cli_args_types import Configuration
from comfy.cmd import folder_paths
from comfy.cmd.folder_paths import init_default_paths
from comfy.component_model.folder_path_types import FolderNames, ModelPaths
from comfy.execution_context import context_folder_names_and_paths


@pytest.fixture()
def clear_folder_paths():
    # Clear the global dictionary before each test to ensure isolation
    with context_folder_names_and_paths(FolderNames()):
        yield


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield tmpdirname


@pytest.fixture
def set_base_dir_t():
    fn = FolderNames()

    def _set_base_dir(base_dir):
        fn.base_paths.clear()
        fn.add_base_path(Path(base_dir))
        init_default_paths(fn, base_paths_from_configuration=False)

    yield _set_base_dir, fn


def test_get_directory_by_type(clear_folder_paths):
    test_dir = "/test/dir"
    folder_paths.set_output_directory(test_dir)
    assert folder_paths.get_directory_by_type("output") == test_dir
    assert folder_paths.get_directory_by_type("invalid") is None


def test_annotated_filepath():
    assert folder_paths.annotated_filepath("test.txt") == ("test.txt", None)
    assert folder_paths.annotated_filepath("test.txt [output]") == ("test.txt", folder_paths.get_output_directory())
    assert folder_paths.annotated_filepath("test.txt [input]") == ("test.txt", folder_paths.get_input_directory())
    assert folder_paths.annotated_filepath("test.txt [temp]") == ("test.txt", folder_paths.get_temp_directory())


def test_get_annotated_filepath():
    default_dir = "/default/dir"
    assert folder_paths.get_annotated_filepath("test.txt", default_dir) == os.path.join(default_dir, "test.txt")
    assert folder_paths.get_annotated_filepath("test.txt [output]") == os.path.join(folder_paths.get_output_directory(), "test.txt")


def test_add_model_folder_path_append(clear_folder_paths):
    folder_paths.add_model_folder_path("test_folder", "/default/path", is_default=True)
    folder_paths.add_model_folder_path("test_folder", "/test/path", is_default=False)
    assert folder_paths.get_folder_paths("test_folder") == ["/default/path", "/test/path"]


def test_add_model_folder_path_insert(clear_folder_paths):
    folder_paths.add_model_folder_path("test_folder", "/test/path", is_default=False)
    folder_paths.add_model_folder_path("test_folder", "/default/path", is_default=True)
    assert folder_paths.get_folder_paths("test_folder") == ["/default/path", "/test/path"]


def test_add_model_folder_path_re_add_existing_default(clear_folder_paths):
    folder_paths.add_model_folder_path("test_folder", "/test/path", is_default=False)
    folder_paths.add_model_folder_path("test_folder", "/old_default/path", is_default=True)
    assert folder_paths.get_folder_paths("test_folder") == ["/old_default/path", "/test/path"]
    folder_paths.add_model_folder_path("test_folder", "/test/path", is_default=True)
    assert folder_paths.get_folder_paths("test_folder") == ["/test/path", "/old_default/path"]


def test_add_model_folder_path_re_add_existing_non_default(clear_folder_paths):
    folder_paths.add_model_folder_path("test_folder", "/test/path", is_default=False)
    folder_paths.add_model_folder_path("test_folder", "/default/path", is_default=True)
    assert folder_paths.get_folder_paths("test_folder") == ["/default/path", "/test/path"]
    folder_paths.add_model_folder_path("test_folder", "/test/path", is_default=False)
    assert folder_paths.get_folder_paths("test_folder") == ["/default/path", "/test/path"]


def test_filter_files_extensions():
    files = ["file1.txt", "file2.jpg", "file3.png", "file4.txt"]
    assert folder_paths.filter_files_extensions(files, [".txt"]) == ["file1.txt", "file4.txt"]
    assert folder_paths.filter_files_extensions(files, [".jpg", ".png"]) == ["file2.jpg", "file3.png"]
    assert folder_paths.filter_files_extensions(files, []) == files


def test_get_filename_list(temp_dir):
    base_path = Path(temp_dir)
    fn = FolderNames(base_paths=[base_path])
    rel_path = Path("test/path")
    fn.add(ModelPaths(["test_folder"], additional_relative_directory_paths=[rel_path], supported_extensions={".txt"}))
    dir_path = base_path / rel_path
    Path.mkdir(dir_path, parents=True, exist_ok=True)
    files = ["file1.txt", "file2.jpg"]

    for file in files:
        Path.touch(dir_path / file, exist_ok=True)

    with context_folder_names_and_paths(fn):
        assert folder_paths.get_filename_list("test_folder") == ["file1.txt"]


def test_get_save_image_path(temp_dir):
    with context_folder_names_and_paths(FolderNames(base_paths=[Path(temp_dir)])):
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path("test", temp_dir, 100, 100)
        assert os.path.samefile(full_output_folder, temp_dir)
        assert filename == "test"
        assert counter == 1
        assert subfolder == ""
        assert filename_prefix == "test"


def test_add_output_path_absolute(temp_dir):
    names = FolderNames()
    config = Configuration()
    config.cwd = str(temp_dir)
    init_default_paths(names, config)
    with context_folder_names_and_paths(names):
        folder_paths.add_model_folder_path("diffusion_models", os.path.join(folder_paths.get_output_directory(), "diffusion_models"))
        mp: ModelPaths = next(names.get_paths("diffusion_models"))
        assert len(mp.additional_absolute_directory_paths) == 0
        assert len(mp.additional_relative_directory_paths) == 1
        assert list(mp.additional_relative_directory_paths)[0] == (Path("output") / "diffusion_models")


def test_base_path_changes(set_base_dir_t):
    test_dir = os.path.abspath("/test/dir")
    set_base_dir, fn = set_base_dir_t
    set_base_dir(test_dir)

    with context_folder_names_and_paths(fn):
        assert str(folder_paths.base_path) == test_dir
        assert str(folder_paths.models_dir) == os.path.join(test_dir, "models")
        assert str(folder_paths.input_directory) == os.path.join(test_dir, "input")
        assert str(folder_paths.output_directory) == os.path.join(test_dir, "output")
        assert str(folder_paths.temp_directory) == os.path.join(test_dir, "temp")
        assert str(folder_paths.user_directory) == os.path.join(test_dir, "user")

        assert os.path.join(test_dir, "custom_nodes") in folder_paths.get_folder_paths("custom_nodes")

        for name in ["checkpoints", "loras", "vae", "configs", "embeddings", "controlnet", "classifiers"]:
            assert folder_paths.get_folder_paths(name)[0] == os.path.join(test_dir, "models", name)


def test_base_path_change_clears_old(set_base_dir_t):
    test_dir = os.path.abspath("/test/dir")
    set_base_dir, fn = set_base_dir_t
    set_base_dir(test_dir)

    with context_folder_names_and_paths(fn):
        assert len(folder_paths.get_folder_paths("custom_nodes")) == 1

        single_model_paths = [
            "checkpoints",
            "loras",
            "vae",
            "clip_vision",
            "style_models",
            "diffusers",
            "vae_approx",
            "gligen",
            "upscale_models",
            "embeddings",
            "hypernetworks",
            "photomaker",
            "classifiers",
        ]
        for name in single_model_paths:
            assert len(folder_paths.get_folder_paths(name)) == 1

        for name in ["controlnet", "diffusion_models", "text_encoders", "configs"]:
            assert len(folder_paths.get_folder_paths(name)) == 2
