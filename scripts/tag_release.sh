#!/usr/bin/env bash
# Create (and optionally push) an annotated git tag matching pyproject.toml version.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="${HOME}/.local/bin:${PATH}"

PUSH=0
DRY_RUN=0
VERSION=""

usage() {
  cat <<'EOF'
Usage: tag_release.sh [OPTIONS] [VERSION]

Create annotated tag vX.Y.Z after release gates pass.
VERSION defaults to pyproject.toml project.version.

Options:
  --dry-run   Validate only; do not create a tag
  --push      Push the tag to origin after creation
  -h, --help  Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --push) PUSH=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h | --help)
      usage
      exit 0
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      VERSION="$1"
      shift
      ;;
  esac
done

if [[ -z "$VERSION" ]]; then
  VERSION="$(python3 scripts/pyproject_tools.py version)"
fi

TAG="v${VERSION}"

echo "LLMIR release tag: ${TAG}"
echo ""

echo "=== Version / changelog gates ==="
pytest tests/test_prepare_release.py -q

python3 scripts/pyproject_tools.py check-alignment "${VERSION}"

TAG_EXISTS=0
if git rev-parse -q --verify "refs/tags/${TAG}" >/dev/null 2>&1; then
  TAG_EXISTS=1
elif git ls-remote --exit-code --tags origin "refs/tags/${TAG}" >/dev/null 2>&1; then
  TAG_EXISTS=1
fi

if [[ "$TAG_EXISTS" == "1" ]]; then
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "DRY RUN: tag ${TAG} already exists (release gates OK)"
    exit 0
  fi
  echo "Tag ${TAG} already exists locally or on origin." >&2
  exit 1
fi

if [[ "$DRY_RUN" != "1" && -n "$(git status --porcelain)" ]]; then
  echo "Working tree is not clean; commit or stash before tagging." >&2
  git status --short >&2
  exit 1
fi

if [[ "$DRY_RUN" == "1" ]]; then
  echo "DRY RUN: would create annotated tag ${TAG}"
  exit 0
fi

git tag -a "${TAG}" -m "Release ${VERSION}"
echo "Created tag ${TAG}"

if [[ "$PUSH" == "1" ]]; then
  git push origin "${TAG}"
  echo "Pushed ${TAG} — check Python package workflow for PyPI publish"
else
  echo "Push with: git push origin ${TAG}"
fi
