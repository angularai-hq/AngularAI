#!/bin/sh
# Puts angularai on your PATH. Does not read or store a key.
set -eu

REPO="git+https://github.com/angularai-hq/AngularAI.git"
CORTEX="${ANGULARAI_CORTEX:-${HOME}/.angularai/default}"

if [ "$(id -u)" -eq 0 ]; then
  printf '%s\n' "Run this as your user. The command installs into your home directory." >&2
  exit 1
fi

printf '%s\n' "angularai" "install" ""

if [ "${ANGULARAI_INSTALL_PLAN:-}" = "1" ]; then
  printf '%s\n' "uv        check" "command   ${REPO}" "cortex    ${CORTEX}"
  exit 0
fi

if ! command -v curl >/dev/null 2>&1; then
  printf '%s\n' "curl is not installed." >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  printf '%s\n' "uv        installing"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  if [ -f "${HOME}/.local/bin/env" ]; then
    # The uv installer writes this file. It is a shell env snippet, not a key.
    # shellcheck disable=SC1091
    . "${HOME}/.local/bin/env"
  fi
fi

if ! command -v uv >/dev/null 2>&1; then
  printf '%s\n' "uv did not install. Install it from https://docs.astral.sh/uv/ and run this again." >&2
  exit 1
fi

BIN="$(uv tool dir --bin)"
export PATH="${BIN}:${PATH}"
printf '%s\n' "uv        ready"
printf '%s\n' "command   installing"
uv tool install --force "${REPO}"

if ! command -v angularai >/dev/null 2>&1; then
  printf '%s\n' "angularai installed, but ${BIN} is not on PATH." >&2
  printf '%s\n' "export PATH=\"${BIN}:\$PATH\"" >&2
  exit 1
fi

if [ ! -d "${CORTEX}" ]; then
  angularai init
else
  printf '%s\n' "cortex    ${CORTEX}"
fi

printf '%s\n' "" "The key stays in the environment." \
  "  export OPENAI_API_KEY=..." \
  "  angularai wire openai --model <model>" \
  "  angularai"
