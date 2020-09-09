"""Pyodi setup."""
from setuptools import find_namespace_packages, setup

MODULE_NAME = "pyodi"
PACKAGE_NAME = "pyodi"
VERSION = "0.0"


def parse_requirements(fname="requirements.txt", with_version=True):
    """Parse the package dependencies listed in a requirements file.

    Args:
        fname (str): path to requirements file
        with_version (bool, default=False): if True include version specs
    Returns:
        List[str]: list of requirements items
    CommandLine:
        python -c "import setup; print(setup.parse_requirements())"
    """
    import re
    import sys
    from os.path import exists

    require_fpath = fname

    def parse_line(line):
        """Parse information from a line in a requirements text file."""
        if line.startswith("-r "):
            # Allow specifying requirements in other files
            target = line.split(" ")[1]
            for info in parse_require_file(target):
                yield info
        else:
            info = {"line": line}
            if line.startswith("-e "):
                info["package"] = line.split("#egg=")[1]
            elif "@git+" in line:
                info["package"] = line
            else:
                # Remove versioning from the package
                pat = "(" + "|".join([">=", "==", ">"]) + ")"
                parts = re.split(pat, line, maxsplit=1)
                parts = [p.strip() for p in parts]

                info["package"] = parts[0]
                if len(parts) > 1:
                    op, rest = parts[1:]
                    if ";" in rest:
                        # Handle platform specific dependencies
                        # http://setuptools.readthedocs.io/en/latest/setuptools.html#declaring-platform-specific-dependencies
                        version, platform_deps = map(str.strip, rest.split(";"))
                        info["platform_deps"] = platform_deps
                    else:
                        version = rest  # NOQA
                    info["version"] = (op, version)
            yield info

    def parse_require_file(fpath):
        with open(fpath, "r") as f:
            for line in f.readlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    for info in parse_line(line):
                        yield info

    def gen_packages_items():
        if exists(require_fpath):
            for info in parse_require_file(require_fpath):
                parts = [info["package"]]
                if with_version and "version" in info:
                    parts.extend(info["version"])
                if not sys.version.startswith("3.4"):
                    # apparently package_deps are broken in 3.4
                    platform_deps = info.get("platform_deps")
                    if platform_deps is not None:
                        parts.append(";" + platform_deps)
                item = "".join(parts)
                yield item

    packages = list(gen_packages_items())
    return packages


if __name__ == "__main__":
    setup(
        name=PACKAGE_NAME,
        version=VERSION,
        description="Object Detection Insights",
        author="Pyodi",
        keywords="computer vision, object detection",
        url="https://github.com/pyodi/pyodi",
        packages=find_namespace_packages(
            include="{}.*".format(MODULE_NAME), exclude=["tests", "logs"]
        ),
        include_package_data=True,
        license="MIT License",
        platforms="any",
        python_requires=">=3.7",
        setup_requires=parse_requirements("requirements/build.txt"),
        tests_require=parse_requirements("requirements/dev.txt"),
        install_requires=parse_requirements("requirements/runtime.txt"),
        extras_require={"all": parse_requirements("requirements.txt")},
        ext_modules=[],
        zip_safe=False,
        entry_points={"console_scripts": ["pyodi = pyodi.cli:app"]},
    )
