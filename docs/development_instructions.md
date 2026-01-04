# Robotics Handbook: Development Instructions

**Project:** Robotics Handbook | **Version:** 1.1 | **Updated:** 2025-12-25

_If outdated, request updated instructions._

## Core Principles

**Factual Accuracy:** All statements must directly correspond to authoritative sources. Prohibit synthetic data, inference, or extrapolation. Restate exact details; paraphrase only if technically precise. Every factual claim requires explicit, verifiable citations.

**Professional Tone:** Maintain technical, measured language resembling specifications. Prohibit anthropomorphic language, personification, or colloquialisms. Prefer passive voice in technical descriptions. Distinguish established facts from speculation: use "is," "has been," "may," "could," or "is expected to" accordingly.

**No Original Research:** Document only existing robotics knowledge, established frameworks, and proven methodologies. No novel algorithms, frameworks, or techniques without explicit sourcing. Original contributions must be clearly marked and reference foundational sources.

## Supported Domains

ROS2, Docker, Physics (dynamics, kinematics, mechanics), Robotics (hardware, actuation, sensing), Embedded systems, Mathematics (linear algebra, calculus, differential equations), Deep learning, Control theory, Machine learning, Autonomous systems, Data-driven design, Reinforcement learning.

## Development Environment

**Base Image:** `rust:1.90-slim-bullseye`

**Required:** VS Code, Dev Container extension, Docker, Docker Compose, Git

**Execution:** Run all development commands (`cargo run`, `mdbook serve`) inside container shell, not host machine.

**Build:** Use `mdbook build` (standard command); `cargo run` serves at `http://localhost:8000`

**Structure:** `/workspace/robotics-handbook/install` (installed files) and `/workspace/robotics-handbook/src` (source files)

## Code Formatting

**File Structure:** Follow existing project structure exactly. Replicate naming conventions and organizational patterns. No new directory hierarchies without approval.

**Comments:** 
- Forbidden patterns: "Add new feature," "Updated this function," "Fix for issue," "Changed by AI," "Modified for better performance"
- Allowed purposes: Explain code purpose, clarify complex logic, group related sections
- Example good comments: "Navigation hint for button controls," "Calculate distance using Euclidean formula," "Thread-safe callback from ROS spinner thread"
- Do not comment self-explanatory code

## Documentation Standards

**Markdown Structure:** Use ATX-style headers (#, ##, ###). Never skip header levels. Maximum three levels in most sections.

**Content:** One concept per section. Paragraphs: 4–6 sentences average. Explicitly connect ideas. Use precise technical terminology.

**Code Blocks:** Use fenced code blocks with language identifiers for syntax highlighting.

**Math:** Use LaTeX only with `\( expression \)` (inline) or `\[ expression \]` (block). Never use $ or $$.

**Lists/Tables:** Unordered lists for non-sequential items; ordered lists for sequences only. Never mix or nest. Use Markdown tables for comparisons (A vs. B), not bulleted lists.

**Citations:** All factual claims cite sources immediately: `statement[id]` with no space. Multiple citations: `statement[1][2][3]`. Inline only; no bibliography.

**Equations:** All math uses LaTeX. Provide context and explain variables before equations. Provide interpretation and application context after.

## Agent Behavior

**Accuracy Mandate:** Never fabricate, infer beyond sources, or present speculation as fact. Trace all explanations to authoritative sources. State explicitly when information is unavailable.

**Source Fidelity:** Paraphrase only if technically precise. Direct quotes use quotation marks and cite immediately. Synthetic examples acceptable only if clearly labeled.

**No Speculative Extensions:** No novel methodologies or frameworks. No invented applications of techniques. Only established practices and research.

**Language:** Professional, technical tone. Use active voice only when agent is relevant; prefer passive for technical descriptions. Avoid hedging unless uncertainty is inherent.

**Error Handling:** Document gaps clearly. Suggest where information might be found. Do not use unverified information.

## Pre-Submission Checklist

- [ ] All factual claims have explicit citations
- [ ] No synthetic data presented as factual
- [ ] Code examples follow style guidelines
- [ ] All imports at file top (Python)
- [ ] All functions/classes have docstrings (Python)
- [ ] Type hints on all functions (Python)
- [ ] No meta-comments or change-description comments
- [ ] Math notation uses LaTeX (`\( \)` or `\[ \]`)
- [ ] Markdown structure uses proper header hierarchy
- [ ] No unused imports or dead code
- [ ] Consistent terminology across handbook
- [ ] Related sections reference each other
- [ ] Code examples compile/execute correctly

**Status:** Active | **Maintainer:** Robotics Handbook Project Team | **Review Cycle:** Quarterly