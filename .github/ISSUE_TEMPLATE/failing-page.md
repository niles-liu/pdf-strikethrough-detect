---
name: Contribute a failing page
about: A page where a strike was missed or a live word was wrongly flagged. These become test cases and training data.
title: "[failing page] "
labels: ["failing-page"]
---

<!-- A missed strike or a false positive is the most useful bug you can file: it turns directly
into a regression test and, once labeled, into training data. Thank you. -->

## The document

- **Can you attach the PDF / image?**  (yes / no — if no, see "Can't share the file" below)
- **Page(s):** <!-- 1-based, e.g. page 3 -->
- **Kind:** <!-- born-digital PDF / scanned PDF / photo or image / .docx -->

## What went wrong

- [ ] A struck (deleted) word was **missed**
- [ ] A live word was **wrongly flagged** as struck
- [ ] Wrong span (partial vs full, wrong characters)
- [ ] Other:

**The word(s):** <!-- the exact text involved -->

**What you expected vs. what happened:**

## How you ran it

```
# the command or code, e.g.
pdf-strikethrough detect doc.pdf --ocr rapidocr --json out.json
```

- **Version:** <!-- pdf-strikethrough --version -->
- **OCR backend (if scanned):** <!-- rapidocr / tesseract / Azure DI / Textract / DocAI / none -->

## Can't share the file?

Export just the affected crops — this leaks no surrounding text and gives us exactly what the
model saw:

```
pdf-strikethrough detect doc.pdf --ocr rapidocr --dump-crops crops_out/
```

Then attach `crops_out/` (the `crops/*.png` + `crops.jsonl`). Even better, set each row's `label`
to `struck` or `clean` first — that makes it labeled training data straight away
(see `training/README.md`).
