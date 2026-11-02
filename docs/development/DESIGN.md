# RAG Lab — design system

## Context

Product workspace for a student presenting retrieval evidence to lecturers in a bright room using a projector. A light, restrained interface keeps source text readable. The established blue accent is retained for actions and selection.

## Color

- Background: `#f5f7fa`; surface: `#ffffff`; secondary surface: `#f8fafc`.
- Primary text: `#182b3d`; supporting text: `#526477`.
- Action: `#0369a1`; hover: `#075985`; selection: `#eaf5fc`.
- Borders: `#dbe3eb`; form boundary: `#bac8d4`.
- Success: `#18704d`; error: `#a32626`; warning: `#795613`.
- Highlighted corpus terms: `#084c74` on `#d7ecfa`.

## Typography

Segoe UI / system sans-serif, loaded locally. Body 15px; input 15px; source excerpt 18px with 1.85 line height; supporting labels 12–13px. Fixed heading scale, 28px desktop and 24px mobile. Use sentence case and Vietnamese labels, retaining English source text.

## Layout

Header, four-stage execution summary, then two-column workspace. Desktop input column 352px with a flexible source inspector. Collapse to one column at 760px. At intermediate widths the process becomes two rows. Technical details use native disclosures. No fixed page height or horizontal body clipping.

## Components

- Blue primary button, neutral secondary button and blue text action.
- Visible field labels, native sliders, minimum 45px primary button height.
- Question presets are buttons that fill the form without submitting.
- Sources are expandable native details with title, document id, chunk id and BM25 score.
- Main excerpt uses a pale blue surface and full border, followed by provenance.
- Reference answer is a separate, collapsed dataset comparison after a completed request.
- Inline errors and retry, dedicated no-result state and reduced-motion skeleton loading.

## Behavior and accessibility

Keep all content usable by keyboard. Ctrl/Command+Enter submits; Enter adds a line. Loading disables conflicting controls. Show measured times and real corpus counts. Changed input shows a stale-result notice. On mobile, completed requests move focus to the result heading. Announce status and clipboard feedback. Respect reduced motion and maintain WCAG AA text contrast.

## Content integrity

Research view: framed-sheet layout (seed 1389653534), blue-grey frame, white sheet, existing Segoe UI/type and blue accent retained. Its main question is whether retrieval changes this answer; three peer answer columns and actual context are the signature element. No sample metrics are invented. At 1100px answers stack; at 680px the scope rail folds into a compact header. Reference labels remain in a separate disclosure; states distinguish missing artifacts, loading, ready, inference failure and stale results.

Label the running system as BM25 + sentence extraction. Keyword coverage is not QA accuracy. Verbatim source text is evidence, not a guarantee of a correct answer. Do not show hard-coded latency, accuracy, or a T5/FAISS execution state.
