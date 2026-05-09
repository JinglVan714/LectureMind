# Timestamp Safety Net Design

## Background

The validation matrix now exposes cross-duration acceptance issues beyond basic verifier success. The current real baseline has two `points_ts_out_of_chapter` failures:

- `BV1NM1tY3Eu5`: one point in the last chapter has `ts=692s`, but the chapter range is `720-857s`.
- `BV1yX4aznE9s`: multiple points are assigned to chapters whose ranges do not contain their timestamps.

The long-video sample also has `chapters_out_of_budget`, but this design intentionally does not address chapter splitting.

## Scope

Implement a deterministic timestamp safety net for chapter points only.

In scope:

- Ensure every point timestamp in final `LectureJSON`/`LectureIR` output is inside its owning chapter range.
- Prefer evidence-preserving relocation by re-locating quote/text within the chapter subtitle window when possible.
- Fall back to the chapter midpoint when no in-chapter evidence match is found.
- Add unit tests for adjacent-chapter timestamp drift.

Out of scope:

- Automatic long-chapter splitting.
- Prompt changes.
- Re-running real model validation unless explicitly needed after implementation.
- Changing duration budgets.

## Design

The safety net should live in the existing normalization layer, not in the renderer or matrix script.

For each point after chapter ranges are known:

1. Read chapter `start` and `end`.
2. If `point.ts` is already within `[start - 1s, end + 1s]`, keep it.
3. If not, try to locate a better timestamp from the point `quote` or `text` within the chapter subtitle window.
4. If no match is found, set `point.ts` to the chapter midpoint.
5. Clamp the final value to `[start, end]` and video duration.

This should be applied after `_fill_chapter_ranges`, because some current failures happen after ranges are filled or extended.

## Error Handling

The safety net must be non-throwing. Missing subtitles, empty quotes, malformed timestamps, or zero-length chapters should fall back to a safe clamped midpoint.

## Testing

Add tests that cover:

- A point whose timestamp is before its chapter range is moved into the chapter.
- A point whose timestamp is after its chapter range is moved into the chapter.
- A point already inside the chapter is unchanged.
- Existing smoke tests remain green.

## Acceptance Criteria

- `pytest tests/test_smoke.py -q` passes.
- `pytest -q` passes.
- The validation matrix can no longer report `points_ts_out_of_chapter` for outputs generated through the updated normalizer.
- Long-video `chapters_out_of_budget` remains visible as a separate acceptance issue.
