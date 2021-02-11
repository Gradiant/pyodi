# Contributing to pyodi

All kinds of contributions are welcome, including but not limited to the following.

- Fixes (typo, bugs)
- New features and components

## Workflow

1. fork and pull the latest pyodi version
2. checkout a new branch (do not use master branch for PRs)
3. commit your changes
4. create a PR

Note
- If you plan to add some new features that involve large changes, it is encouraged to open an issue for discussion first.


## Code style

### Python
We adopt [PEP8](https://www.python.org/dev/peps/pep-0008/) as the preferred code style.

We use the following tools for linting and formatting:
- [flake8](http://flake8.pycqa.org/en/latest/): linter
- [black](https://github.com/psf/black): formatter
- [isort](https://github.com/timothycrosley/isort): sort imports

Style configurations of black and isort can be found in [pyproject.toml](../.pyproject.toml).

We use [pre-commit hook](https://pre-commit.com/) that checks and formats for `flake8`, `yapf`, `isort`,
 fixes `end-of-files`, automatically on every commit.
The config for a pre-commit hook is stored in [.pre-commit-config](../.pre-commit-config.yaml).

After you clone the repository and installed pyodi (if you use the `dev` extras require `pre-commit` and all the necessary packages for development will be installed), you will need to install initialize pre-commit hook.

```bash
pip install -e .[dev]
pre-commit install
```

After this on every commit check code linters and formatter will be enforced.
