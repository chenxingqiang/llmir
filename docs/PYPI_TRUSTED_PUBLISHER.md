# PyPI Trusted Publishing (GitHub Actions)

`v0.2.2` CI **test / lint / build** pass, but **publish** may fail with:

```text
invalid-publisher: valid token, but no corresponding publisher
```

This means GitHub OIDC tokens reach PyPI, but the **trusted publisher** is not
configured for this repository.

## Fix (maintainer, one-time)

1. Log in to https://pypi.org/manage/project/llmir/settings/publishing/
2. Add a **pending publisher** (or edit existing):
   - **PyPI project**: `llmir`
   - **Owner**: `chenxingqiang`
   - **Repository**: `llmir`
   - **Workflow**: `python-package.yml`
   - **Environment**: `pypi` (matches workflow `environment: name: pypi`)
3. Merge/save on PyPI, then re-run publish:
   - Re-push the same tag (delete remote tag first if needed), **or**
   - GitHub Actions → **Python Package** → re-run failed `publish` job on tag `v*`

## Verify after publish

```bash
python3 scripts/verify_pypi_release.py
cat IEEE-conference/benchmarks/pypi_release_status.json
```

Expected: `"status": "published"`, `pypi_version` equals `local_version`.

## Manual fallback (API token)

If trusted publishing is blocked, upload from CI artifacts:

```bash
bash scripts/prepare_release.sh
twine upload dist/*   # TWINE_USERNAME=__token__ TWINE_PASSWORD=pypi-...
python3 scripts/verify_pypi_release.py
```

See also: `docs/PYPI_RELEASE_CHECKLIST.md`.
