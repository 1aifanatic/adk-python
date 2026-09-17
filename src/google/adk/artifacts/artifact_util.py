# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utility functions for handling artifact URIs."""

from __future__ import annotations

import re
from typing import NamedTuple

from google.genai import types

from ..errors import input_validation_error
from .base_artifact_service import MediaFrame


class ParsedArtifactUri(NamedTuple):
  """The result of parsing an artifact URI."""

  app_name: str
  user_id: str
  session_id: str | None
  filename: str
  version: int


_WINDOWS_DRIVE_RE = re.compile(r"[A-Za-z]:")

_RESERVED_PATH_SEGMENTS = frozenset({
    "apps",
    "users",
    "sessions",
    "artifacts",
    "versions",
})
_RESERVED_SEGMENTS_LOOKAHEAD = (
    rf"(?!/(?:{'|'.join(sorted(_RESERVED_PATH_SEGMENTS))})/)"
)
_PATH_SEGMENT_PATTERN = rf"(?:{_RESERVED_SEGMENTS_LOOKAHEAD}.)+?"

_SESSION_SCOPED_ARTIFACT_URI_RE = re.compile(
    rf"artifact://apps/({_PATH_SEGMENT_PATTERN})/users/({_PATH_SEGMENT_PATTERN})/sessions/({_PATH_SEGMENT_PATTERN})/artifacts/(.+)/versions/(\d+)"
)
_USER_SCOPED_ARTIFACT_URI_RE = re.compile(
    rf"artifact://apps/({_PATH_SEGMENT_PATTERN})/users/({_PATH_SEGMENT_PATTERN})/artifacts/(.+)/versions/(\d+)"
)

# Layout shared by every backend that stores media frame collections.
FRAMES_DIR_NAME = "frames"
MEDIA_COLLECTION_TYPE = "video_frame_sequence"
DEFAULT_FRAME_MIME_TYPE = "image/jpeg"
DEFAULT_FRAME_EXTENSION = "jpeg"

# Scope marker every backend strips before a name becomes a stored path. Each
# backend owns its own handling of it; this copy exists so the shared
# validation below sees the same name storage will.
_USER_NAMESPACE_PREFIX = "user:"

# A frame extension becomes a path segment, so it is held to the same standard
# as the caller-supplied identifiers `validate_path_segment` guards: no
# separators, no traversal, no null bytes. Restricting to the characters real
# subtypes use is simpler to reason about than enumerating what to reject.
_FRAME_EXTENSION_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9+._-]*")


def frame_extension_from_mime_type(mime_type: str | None) -> str:
  """Derives the filename extension for a media frame from its mime type.

  The mime type reaches the artifact services from a live model stream, so it
  is caller-supplied data that ends up inside a path. Anything that is not a
  plausible subtype -- including values carrying "/" or "\\" separators or ".."
  traversal segments -- falls back to the default rather than raising, because
  an odd mime type should not fail a whole frame batch.

  Args:
    mime_type: The blob mime type, e.g. "image/jpeg" or "image/png;foo=bar".

  Returns:
    A path-safe extension, defaulting to "jpeg".
  """
  subtype = (mime_type or "").split("/")[-1].split(";")[0].strip()
  if not _FRAME_EXTENSION_RE.fullmatch(subtype):
    return DEFAULT_FRAME_EXTENSION
  return subtype


def frame_file_name(index: int, mime_type: str | None) -> str:
  """Returns the `frames/`-relative filename for one frame."""
  return f"frame_{index:04d}.{frame_extension_from_mime_type(mime_type)}"


def validate_media_collection_name(collection_name: str) -> None:
  """Rejects a collection name that collides with the frames subdirectory.

  `FileArtifactService` stages a version's preview file and its `frames/`
  subdirectory as siblings, so a collection whose own final segment is "frames"
  makes them the same path and the save dies partway through with
  `IsADirectoryError`. The flat-namespace backends have no such collision, but
  they reject the name too: a caller that can write a collection on one backend
  and not another has no portable contract to program against, and the failure
  only shows up after switching backends.

  The name is normalized the way the backends normalize it before it reaches
  storage -- the `user:` scope marker is not part of the stored path, and the
  file backend strips surrounding whitespace -- so `user:frames` is caught too.

  Args:
    collection_name: The caller-supplied media collection name.

  Raises:
    InputValidationError: If the name's final path segment is "frames".
  """
  stripped = collection_name.removeprefix(_USER_NAMESPACE_PREFIX).strip()
  if stripped.rpartition("/")[2].casefold() == FRAMES_DIR_NAME:
    raise input_validation_error.InputValidationError(
        f"Collection filename {collection_name!r} is reserved for media frame"
        " storage."
    )


def validate_frame_timestamps(
    frames: list[MediaFrame],
) -> None:
  """Rejects a frame batch whose timestamps run backwards.

  Every backend takes `durationMs` from the first and last frame and each
  frame's `offsetMs` from the first, so an out-of-order batch does not fail --
  it silently persists a negative duration, negative offsets, and a meaningless
  `estimatedFps`. Callers buffer frames as they arrive, so out-of-order input
  means the caller has a bug worth surfacing rather than recording.

  Equal timestamps are allowed: frames captured within the same clock tick are
  legitimate and yield a zero offset delta.

  Args:
    frames: The `MediaFrame` batch about to be written.

  Raises:
    InputValidationError: If any frame predates the frame before it.
  """
  for idx in range(1, len(frames)):
    previous_ts = frames[idx - 1].timestamp
    current_ts = frames[idx].timestamp
    if current_ts < previous_ts:
      raise input_validation_error.InputValidationError(
          f"Frame timestamps must be non-decreasing, but frame {idx} at"
          f" {current_ts} precedes frame {idx - 1} at {previous_ts}."
      )


def parse_artifact_uri(uri: str) -> ParsedArtifactUri | None:
  """Parses an artifact URI.

  Args:
      uri: The artifact URI to parse.

  Returns:
      A ParsedArtifactUri if parsing is successful, None otherwise.
  """
  if not uri or not uri.startswith("artifact://"):
    return None

  match = _SESSION_SCOPED_ARTIFACT_URI_RE.fullmatch(uri)
  if match:
    return ParsedArtifactUri(
        app_name=match.group(1),
        user_id=match.group(2),
        session_id=match.group(3),
        filename=match.group(4),
        version=int(match.group(5)),
    )

  match = _USER_SCOPED_ARTIFACT_URI_RE.fullmatch(uri)
  if match:
    return ParsedArtifactUri(
        app_name=match.group(1),
        user_id=match.group(2),
        session_id=None,
        filename=match.group(3),
        version=int(match.group(4)),
    )

  return None


def get_artifact_uri(
    app_name: str,
    user_id: str,
    filename: str,
    version: int,
    session_id: str | None = None,
) -> str:
  """Constructs an artifact URI.

  Args:
      app_name: The name of the application.
      user_id: The ID of the user.
      filename: The name of the artifact file.
      version: The version of the artifact.
      session_id: The ID of the session.

  Returns:
      The constructed artifact URI.
  """
  if session_id:
    return f"artifact://apps/{app_name}/users/{user_id}/sessions/{session_id}/artifacts/{filename}/versions/{version}"
  else:
    return f"artifact://apps/{app_name}/users/{user_id}/artifacts/{filename}/versions/{version}"


def is_artifact_ref(artifact: types.Part) -> bool:
  """Checks if an artifact part is an artifact reference.

  Args:
      artifact: The artifact part to check.

  Returns:
      True if the artifact part is an artifact reference, False otherwise.
  """
  return bool(
      artifact.file_data
      and artifact.file_data.file_uri
      and artifact.file_data.file_uri.startswith("artifact://")
  )


def validate_artifact_reference_scope(
    *,
    app_name: str,
    user_id: str,
    session_id: str | None,
    parsed_uri: ParsedArtifactUri,
) -> None:
  """Ensures artifact references cannot escape the caller's scope."""
  if parsed_uri.app_name != app_name or parsed_uri.user_id != user_id:
    raise input_validation_error.InputValidationError(
        "Artifact references must stay within the same app and user scope."
    )
  if parsed_uri.session_id is not None and parsed_uri.session_id != session_id:
    raise input_validation_error.InputValidationError(
        "Session-scoped artifact references must stay within the same"
        " session scope."
    )


def _is_drive_qualified(value: str) -> bool:
  """Checks whether a value starts with a Windows drive letter such as ``C:``."""
  return _WINDOWS_DRIVE_RE.match(value) is not None


def _validate_session_id_for_flat_storage(session_id: str) -> None:
  """Validates a session_id used by flat storage artifact backends.

  In addition to the checks in `validate_path_segment`, rejects values whose
  first path segment is the reserved value "user". Backends that lay out
  session-scoped and user-scoped artifacts in the same flat namespace
  (in-memory, GCS) use that exact string as a reserved segment marking
  user-scoped artifacts, so a session starting with "user" would silently write
  into -- and read out of -- that reserved namespace instead of its own.

  Args:
    session_id: The caller-supplied session id.

  Raises:
    InputValidationError: If `session_id` fails `validate_path_segment`, or has
      the reserved value "user" as its first path segment.
  """
  validate_path_segment(session_id, "session_id")
  if session_id.replace("\\", "/").split("/")[0] == "user":
    raise input_validation_error.InputValidationError(
        "session_id must not be or start with the reserved value 'user'."
    )


def validate_path_segment(value: str, field_name: str) -> None:
  """Rejects values that could alter the constructed path.

  Args:
    value: The caller-supplied identifier (e.g. user_id or session_id).
    field_name: Human-readable name used in the error message.

  Raises:
    InputValidationError: If the value contains traversal segments, null bytes,
      is an absolute path / starts with a slash, is drive-qualified, or contains
      slashes along with reserved path segments.
  """
  if not value:
    raise input_validation_error.InputValidationError(
        f"{field_name} must not be empty."
    )
  if "\x00" in value:
    raise input_validation_error.InputValidationError(
        f"{field_name} must not contain null bytes."
    )
  if isinstance(value, str) and (
      value.startswith("/") or value.startswith("\\")
  ):
    raise input_validation_error.InputValidationError(
        f"{field_name} {value!r} must not be an absolute path or start with a"
        " slash."
    )
  if isinstance(value, str) and _is_drive_qualified(value):
    raise input_validation_error.InputValidationError(
        f"{field_name} {value!r} must not be drive-qualified."
    )
  if value in (".", "..") or ".." in value.replace("\\", "/").split("/"):
    raise input_validation_error.InputValidationError(
        f"{field_name} {value!r} must not contain traversal segments."
    )
  if isinstance(value, str) and ("/" in value or "\\" in value):
    segments = {
        segment.casefold() for segment in value.replace("\\", "/").split("/")
    }
    if not _RESERVED_PATH_SEGMENTS.isdisjoint(segments):
      raise input_validation_error.InputValidationError(
          f"{field_name} {value!r} must not contain reserved path segments."
      )
