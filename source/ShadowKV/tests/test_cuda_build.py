from pathlib import Path


from cuda_build import get_cuda_include_dirs


def test_cuda_component_headers_are_added_when_toolkit_header_is_missing(tmp_path: Path):
    cuda_home = tmp_path / "cuda"
    toolkit_include = cuda_home / "include"
    toolkit_include.mkdir(parents=True)

    site_packages = tmp_path / "site-packages"
    cusparse_include = site_packages / "nvidia" / "cusparse" / "include"
    cusparse_include.mkdir(parents=True)
    (cusparse_include / "cusparse.h").touch()

    include_dirs = get_cuda_include_dirs(
        cuda_home=cuda_home,
        site_packages=[site_packages],
    )

    assert toolkit_include in include_dirs
    assert cusparse_include in include_dirs
