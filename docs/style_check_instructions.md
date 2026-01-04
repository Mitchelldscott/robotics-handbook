# Robotics Handbook: Style & Hierarchy Guide

## **1. Structural Architecture**

The content implies two distinct file archetypes: **The Index (Overview)** and
**The Chapter (Deep Dive)**. All files must conform to one of these two
structures.

### **Type A: Index/Overview Files (`index.md`)**

Used for high-level subject grouping and curriculum outlining.

1. **H1 Title:** Broad Subject Area (e.g., "Mathematics for Autonomous
   Systems").
2. **Introduction:** 1–2 paragraphs providing a high-level motivation connecting
   the math to robotics applications.
3. **Visual (Optional):** Mermaid diagram showing domain relationships.
4. **H2 Outline:**

- **H3 Subject Heading:** (e.g., "Dynamical Systems").
- **Syllabus Blockquote:** Immediately following the H3. A blockquote (`>`)
  listing the specific sub-topics/keywords covered in that module.
- **Context Paragraph:** A brief justification of why this theory matters to a
  roboticist.
- **Project List:** Bullet points where the item start is **Bolded** (e.g.,
  `* **Project Name**: Description...`).

### **Type B: Technical Chapters (`dynamical-systems.md`)**

Used for specific theoretical instruction.

1. **H1 Title:** Specific Subject Name (e.g., "Dynamical Systems").
2. **Epigraph/Definition:**

- Must start with a Blockquote (`>`) containing a citation or authoritative
  definition.
- Followed immediately by a high-level summary of the framework (e.g.,
  "Dynamical systems theory concerns...").

3. **Motivation:** A paragraph posing central questions (e.g., "What are the
   equilibrium points?").
4. **H2 Major Section:** (e.g., "1. Modeling Fundamentals").
5. **H3 Sub-topic:** (e.g., "Ordinary Differential Equations").
6. **H4 Specific Concept:** (e.g., "Linear ODEs").

- _Constraint:_ Do not exceed H4 depth unless mathematically necessary for a
  derivation steps (H5).

---

## **2. Syntactic & Formatting Standards**

### **Typography & Emphasis**

- **Bold (`**text**`):** STRICTLY reserved for **defining new terms** (e.g.,
  "**State vector**:", "**Poles** and **zeros**") or highlighting key list
  headers.
- **Italics (`_text_`):** Used for variables in standard text (if not using
  LaTeX) or emphasis within a definition.
- **Blockquotes (`>`):** Used for:
- Epigraphs/Quotes at the start of a document.
- Syllabus lists in Index files.
- Vital definitions or theorems (e.g., Observability definition).

### **Mathematical Notation (LaTeX)**

- **Delimiters:**
- Inline math: `\\( ... \\)`
- Display/Block math: `\\[ ... \\]`

- **Variables:** Vectors and matrices must be bolded (e.g., `\mathbf{x}`, `A`).
- **Spacing:** Do not use code blocks for math. Use standard Markdown text flow
  with LaTeX delimiters.

### **Visuals & Tables**

- **Tables:** Use Markdown tables for classification logic (e.g., "Type |
  Condition | Behavior").
- **Diagrams:** Mermaid graphs are preferred for relationships/flow.
- **Plots:** Reference standard plot types (Phase Portraits, Bode Plots) in H3
  or H4 headers.

---

## **3. Tone & Vocabulary**

- **Voice:** Authoritative, academic, yet practical.
- _Bad:_ "I think this is important because..."
- _Good:_ "This framework applies to both continuous-time..."

- **Wordiness:** High density. Avoid fluff. Sentences should be declarative.
- **Perspective:** Third-person objective.
- **Robotics Context:** Every theoretical section must bridge back to physical
  utility (e.g., "The point of modeling is to capture... behavior of a physical
  system").

---

## **4. Reference System**

- **In-text:** Use bracketed keys `[citation_key]` or ``.
- **Bibliography:**
- Located at the very bottom of the file under `## References`.
- Format: Raw BibTeX inside a code block (````bibtex`).

---

## **5. Automated Review Checklist**

If you are writing a linter, these logic gates determine a "Pass":

1. **Header Continuity:** Does H1 exist? Is it the first line?
2. **Intro Block:** Does the content immediately following H1 contain a
   Blockquote (`>`)?
3. **Math Check:** Are all `[` and `(` brackets used for LaTeX properly escaped
   with double backslashes (`\\[`, `\\(`)?
4. **Hierarchy Check:** Are H3s nested under H2s? (No skipping levels).
5. **Term Definition:** Does the text contain `**Term**: Definition` patterns?
6. **BibTeX Presence:** Does the file end with a `## References` section
   containing a `bibtex` block?
7. **Tone Check**: Does the file use the right vocabulary and perspective.
8. **Size Check**: Are any sections of text unnecessary or is the file too long?
