# Release assets

No large asset is staged for GitHub Release or Git LFS. Files above the GitHub 50 MB warning threshold are listed with byte counts and SHA-256 in `RELEASE_ASSETS_TODO.md`. They are excluded because they contain third-party benchmark text, complete request/passage strings, private intermediate inputs, or duplicate diagnostic bundles. Sanitized ID/score projections needed for the released analyses are already in this repository.
