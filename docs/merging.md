# Merging Upstream Changes

This document covers synchronization tasks needed when merging upstream ComfyUI changes.

## CLI Arguments

When upstream adds new CLI arguments to `comfy/cli_args.py`, you must also update `comfy/cli_args_types.py`:

1. **Docstring** - Add the new argument's documentation to the `Configuration` class docstring
2. **__init__** - Add the new attribute with its default value to `Configuration.__init__`

### Example

If upstream adds:
```python
parser.add_argument("--disable-assets-autoscan", action="store_true", help="Disable asset scanning...")
```

Then add to `cli_args_types.py`:

In the docstring:
```python
disable_assets_autoscan (bool): Disable asset scanning on startup for database synchronization.
```

In `__init__`:
```python
self.disable_assets_autoscan: bool = False
```

### Quick Check

After merging, diff the argument names:
```bash
grep -oP '(?<=add_argument\(")[^"]+' comfy/cli_args.py | sed 's/^--//' | sed 's/-/_/g' | sort > /tmp/args.txt
grep -oP '(?<=self\.)[a-z_]+(?=:)' comfy/cli_args_types.py | sort > /tmp/types.txt
diff /tmp/args.txt /tmp/types.txt
```

## Version String

Upstream uses `comfyui_version.py` at the repository root. We deleted this file and moved the version string to `comfy/__init__.py`.

When merging, accept our deletion of `comfyui_version.py` and update the `__version__` in `comfy/__init__.py` instead.

## Requirements

Upstream uses `requirements.txt` at the repository root. We deleted this file and moved all dependencies to `pyproject.toml`.

When merging, accept our deletion of `requirements.txt` and update version minimums in `pyproject.toml` instead. Key packages to watch:

- `comfyui-frontend-package`
- `comfyui-workflow-templates`
- `comfyui-embedded-docs`
- `comfy_kitchen` (our addition, not in upstream)

## Package Init Files

Upstream sometimes adds new Python packages without `__init__.py` files. After merging, check for missing init files and add empty ones where needed to make them proper Python packages.

## Directory Structure

Our fork moves some top-level directories into `comfy/`. When upstream adds or modifies files in these directories:

1. First, merge the upstream changes to the top-level location
2. In a separate commit, `git mv` the files to the correct location

Example: `app/assets` → `comfy/app/assets`

This two-step approach keeps git history cleaner and makes conflicts easier to resolve.

## Import Fixes

After moving directories, fix absolute imports to use relative imports. Upstream uses absolute imports like:

```python
import app.assets.manager as manager
from app.database.db import create_session
from app.assets.helpers import some_function
```

Convert these to relative imports based on file location:

```python
from .. import manager
from ..database.db import create_session
from .helpers import some_function
```

Reference:
- `.` = current package
- `..` = parent package
- `...` = grandparent package
