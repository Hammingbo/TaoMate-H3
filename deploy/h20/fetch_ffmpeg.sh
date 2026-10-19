#!/usr/bin/env bash
# BtbN builds are linked from https://ffmpeg.org/download.html.
# Download on a host with Internet access; execute on the newer GPU host.
set -euo pipefail
DEST="${1:?Pass a shared destination directory}"
mkdir -p "$DEST"
cd "$DEST"
exec 9>ffmpeg-download.lock
flock -n 9 || { echo 'A media download is already running'; exit 1; }
ARCHIVE=ffmpeg-n8.1-latest-linux64-gpl-8.1.tar.xz
BASE_URL=https://github.com/BtbN/FFmpeg-Builds/releases/download/latest
curl -fL --retry 2 --connect-timeout 10 --max-time 240 "$BASE_URL/$ARCHIVE" -o "$ARCHIVE"
curl -fL --retry 2 --max-time 30 "$BASE_URL/checksums.sha256" -o ffmpeg-checksums.sha256
grep " $ARCHIVE$" ffmpeg-checksums.sha256 | sha256sum -c -
mkdir -p media-dist
tar -xJf "$ARCHIVE" --no-same-owner --strip-components=1 -C media-dist
printf 'MEDIA_DIST_READY: %s/media-dist/bin\n' "$PWD"
