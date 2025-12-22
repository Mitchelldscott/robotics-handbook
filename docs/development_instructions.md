# Robotics Handbook: Instruction File for Formatting and Language Rules

## Overview

This instruction file establishes the definitive standards for all content, code, and documentation within the Robotics Handbook project. All contributors and documentation agents must adhere strictly to these rules. Deviations from these guidelines are not acceptable without explicit project-level approval and rule updates to this instruction file.

**Project Name:** Robotics Handbook  
**Project Description:** A handbook for robotics, covering mathematics, algorithms, embedded systems, and resources.  
**Instruction Version:** 1.0  
**Last Updated:** 2025-12-19

---

## 1. Core Principles

### 1.1 Factual Accuracy and Source Fidelity

- **Primary Rule:** All statements, explanations, and technical details must directly correspond to authoritative sources. Do not infer, extrapolate, or synthesize information beyond what source material explicitly states.
- **Synthetic Data Prohibition:** Under no circumstances shall synthetic, simulated, or hypothetical data be presented as factual. If data cannot be sourced from authoritative references, this must be clearly stated.
- **Exact Details Requirement:** Restate exact details from sources. Paraphrasing is permitted only when it preserves technical accuracy and the original meaning.
- **Source Citation Mandate:** Every factual claim must be traceable to a source. Citations must be explicit and verifiable.

### 1.2 Professional and Factual Tone

- **Tone Standard:** All documentation shall maintain a professional, measured, and factual tone resembling technical robotics specifications rather than conversational guidance.
- **Anthropomorphic Language Prohibition:** Avoid anthropomorphic descriptions, personification, or colloquialisms. Use precise technical terminology consistently.
- **Passive Voice Preference:** Prefer passive voice in technical descriptions to emphasize the action and result rather than the agent.
- **Certainty vs. Speculation:** Clearly distinguish between established facts and speculative or theoretical content. Use precise language: "is," "has been," "may," "could," or "is expected to" depending on certainty level.

### 1.3 No Original Research or Invention

- The handbook documents existing robotics knowledge, established frameworks, and proven methodologies.
- Contributors shall not introduce novel algorithms, frameworks, or techniques without explicit sourcing and verification.
- Original contributions must be clearly marked as extensions or implementations and must still reference foundational sources.

---

## 2. Supported Languages and Technologies

### 2.1 Primary Languages

The handbook supports code examples and content in the following languages. All code must be formatted according to language-specific standards defined in this file.

| Language | Role | Formatter/Linter | Notes |
|----------|------|------------------|-------|
| Markdown | Documentation | Markdown standards (GitHub-flavored) | Primary documentation format |
| Julia | Scientific computing, mathematics | JuliaFormatter | For mathematical algorithms and simulations |
| Python | ROS2 nodes, scripting, ML | black, flake8, isort | Primary systems language for ROS2 |
| C++ | Embedded systems, performance-critical code | clang-format (project config) | For high-performance robotics code |
| Rust | Systems code, safety-critical components | rustfmt | For memory-safe robotics applications |
| Bash | Build scripts, automation | shellcheck (recommended) | For development and deployment scripts |
| LaTeX | Mathematical expressions, technical documentation | texlive | For complex mathematical notation |
| CSS | Web documentation styling | Prettier (recommended) | For web-based handbook viewing |
| HTML | Web documentation structure | HTML5 standards | For web-based handbook viewing |

### 2.2 Primary Domains

Content shall address the following robotics domains with equal rigor and factual precision:

- ROS2 (Robot Operating System 2)
- Docker and containerization
- Physics (dynamics, kinematics, mechanics)
- Robotics (hardware, actuation, sensing)
- Embedded systems (microcontrollers, firmware)
- Mathematics (linear algebra, calculus, differential equations)
- Deep learning (neural networks, perception)
- Control theory (feedback control, trajectory planning)
- Machine learning (supervised, unsupervised, reinforcement learning)
- Autonomous systems (planning, navigation, decision-making)
- Data-driven design (system identification, optimization)
- Reinforcement learning (MDP, policy optimization)

---

## 3. Development Environment Standards

### 3.1 Docker and Dev Containers

- **Base Image:** `rust:1.90-slim-bullseye` is the standard base image for the development environment.
- **Container Usage:** VS Code must be run inside the Docker container via the Dev Container extension.
- **Command Execution:** All development commands such as `cargo run` and `mdbook serve` must be executed inside the container shell, not on the host machine.
- **Host Requirements:**
    - VS Code installed
    - Dev Container extension installed
    - Docker installed
    - Docker Compose installed
    - Git installed

### 3.2 Build Commands

- **Preferred Build Command:** `mdbook build` is the standard command for building the handbook.
- **Serving the Handbook:** `cargo run` serves the handbook on `http://localhost:8000`.
- **Never deviate from these commands without explicit project approval.**

### 3.3 Critical File Structure

The following directory structure and files are critical to the project:

- `/workspace/robotics-handbook/install` – Installation and setup scripts
- `/workspace/robotics-handbook/src` – Source code and documentation content

---

## 4. Code Formatting and Style Guidelines

### 4.1 General Coding Principles

#### 4.1.1 Stateless Behavior

- Code shall be structured as **modular ROS2 nodes with clear separation of concerns**.
- **State Management Rule:** Avoid maintaining state inside node objects. Instead, use periodic state updates via ROS2 topics and services.
- **Rationale:** This ensures robustness and correct recovery after failures or node restarts.

#### 4.1.2 Code Reuse

- Prefer simple functions and classes over complex abstractions.
- If code is reused across multiple modules, move it into shared utilities or common files.
- Follow existing project structure and naming conventions when creating shared utilities.

#### 4.1.3 File Structure

- New files must follow the existing project structure exactly.
- Naming conventions and organizational patterns already established in the repository must be replicated.
- Do not introduce new directory hierarchies or naming schemes without explicit approval.

### 4.2 Comments and Docstrings

#### 4.2.1 Meta-Comments Prohibition

The following comment patterns are **strictly forbidden**:

- "Add new feature"
- "Updated this function"
- "Fix for issue"
- "Changed by AI"
- "Modified for better performance"

These comments provide no technical value and obscure the actual purpose of the code.

#### 4.2.2 Allowed Comment Purpose

Comments and docstrings must serve only these purposes:

- **Explain code purpose:** Clarify what the code is designed to accomplish.
- **Clarify complex logic:** Explain non-obvious algorithms or intricate control flow.
- **Organizational grouping:** Use comments to group related code sections for readability.

#### 4.2.3 Comment Style Rules

- Use organizational comments to group related code sections.
- Comment complex algorithms and non-obvious logic only.
- Do not comment self-explanatory code.
- Apply the same principles to docstrings: describe what the code does, not why it changed.

**Examples of Good Comments:**

- "Navigation hint for button controls."
- "Calculate distance using Euclidean formula."
- "Thread-safe callback from ROS spinner thread."

#### 4.2.4 Docstring Conventions

- **Requirement:** All new functions and classes must have docstrings.
- **Update Rule:** Docstrings must be updated whenever the function or class is modified.
- **Format:** Google style + PEP 257 conventions.
- **Structure:**
    - First line: Concise one-line summary ending with a period.
    - Blank line after summary.
    - Detailed description (if necessary).
    - Arguments section (if applicable).
    - Returns section (if applicable).
    - Raises section (if applicable).

### 4.3 Import Statements

#### 4.3.1 Import Location Rule

- **All import statements must be at the top of the file only.**
- **Forbidden locations for imports:**
    - Inside functions
    - Inside methods
    - Inside class bodies (non-top-level)
    - In any inline or local scope

#### 4.3.2 Import Workflow

Before adding any import statement:

1. Read lines 1–30 of the target file to check existing imports.
2. Verify that the import is not already present at the top of the file.
3. Add the new import only to the top-level imports section, before any function or class definitions.

#### 4.3.3 Inline Import Cleanup

- Inline imports found inside functions are incorrect and must be moved to the top of the file.
- Do not replicate incorrect import patterns from existing code.
- **Assumption:** Assume all imports build and resolve correctly; do not add speculative imports.

### 4.4 Python Style Guidelines

#### 4.4.1 Formatting and Linting

- **Formatter:** Use `black` for all Python code.
- **Type Hints:** Type hints are required for all functions and class methods.
- **Unused Imports:** No unused imports are permitted. Remove all unused imports before submission.
- **Configuration Files:** Refer to `.flake8` and `.isort.cfg` in the config folder for style guidance.

#### 4.4.2 Constants and Magic Numbers

- **Magic Number Rule:** Replace all magic numbers and strings with named constants.
- **Placement:** Define all constants at the top of the file, after imports.
- **Naming Convention:** Use `UPPER_SNAKE_CASE` for all constant names.

**Example:**

```python
# Correct
ROBOT_MAX_VELOCITY = 1.5  # m/s
SAFETY_TIMEOUT = 5.0      # seconds

def move_robot(velocity: float) -> None:
    if velocity > ROBOT_MAX_VELOCITY:
        velocity = ROBOT_MAX_VELOCITY
```

#### 4.4.3 Class Initialization

- Initialize values inside the class definition where applicable, not only within `__init__`.
- This improves code clarity and provides default values for documentation purposes.

#### 4.4.4 ROS2 and Common Utilities

**tempoutils Package Usage:**

The `tempoutils` package provides standardized utilities for ROS2 development. Use these utilities exclusively:

- **QoS Profile:** Use `DEFAULT_QOS` from `tempoutils.util` for all ROS2 subscribers and publishers.
- **Time Functions:** Use functions from `tempoutils.time` for time conversions and arithmetic operations.
- **Mathematics:** Use helper functions from `tempoutils.math` where applicable.
- **DateTime:** Use utilities from `tempoutils.datetime_utils` and prefer UTC for all system times.

#### 4.4.5 Time Handling

- **Inside ROS2 Nodes:** Use ROS2 time (via `rclpy` clock) to support simulated time in simulation environments.
- **System Time:** Use UTC consistently for system times. Convert external timezones via `tempoutils.datetime_utils`.

#### 4.4.6 Launch Files and Configuration

- All new features and nodes must be configurable via launch file arguments.
- Configuration shall not be hardcoded into Python files.

### 4.5 C/C++ Style Guidelines

#### 4.5.1 Formatting

- **Configuration File:** Follow the `.clang-format` file in the config folder.
- **Project-Specific Style:** Adhere to all C/C++ style conventions defined in the repository config files.

#### 4.5.2 Header Guard

- Use `#pragma once` for all header files.

#### 4.5.3 Visual Separators

- **Enabled:** Visual separators improve code readability and must be used.
- **Separator Character:** Use the forward slash `/` character.
- **Separator Length:** 75 characters.
- **Usage Locations:**
    - Between classes and structs in `.h` (header) files.
    - Between functions in `.cpp` (implementation) files.

**Example:**

```cpp
class RobotController {
public:
    void update();
private:
    float velocity_;
};

///////////////////////////////////////////////////////////////////////////////

class SensorFusion {
public:
    void process_sensor_data();
};
```

### 4.6 Rust Style Guidelines

- Use `rustfmt` for all Rust code formatting.
- Follow Rust naming conventions: `snake_case` for functions and variables, `PascalCase` for types and traits.
- All public APIs must have documentation comments.

### 4.7 Bash and Shell Scripts

- Use `shellcheck` for linting shell scripts (recommended).
- Include shebang lines: `#!/bin/bash` at the top of all shell scripts.
- Document complex shell operations with comments.

---

## 5. Documentation Standards

### 5.1 Markdown Documentation

#### 5.1.1 Structure

- Use ATX-style headers (`#`, `##`, `###`, etc.) for document structure.
- Do not skip header levels. Use hierarchical structure consistently.
- Maximum of three levels of headers in most sections.

#### 5.1.2 Content Rules

- **One concept per section:** Each section should address a single, well-defined concept.
- **Paragraph length:** Paragraphs should be 4–6 sentences on average.
- **Connections:** Explicitly connect ideas across paragraphs and sections to maintain narrative flow.
- **Technical precision:** Use precise technical terminology and avoid ambiguous language.

#### 5.1.3 Code Blocks

- Use fenced code blocks with language identifiers for syntax highlighting.
- Example:
  ````markdown
  ```python
  def calculate_distance(x1, y1, x2, y2):
      return ((x2 - x1)**2 + (y2 - y1)**2)**0.5
  ```
  ````

#### 5.1.4 Mathematical Notation

- Use LaTeX notation for all mathematical expressions.
- Inline math: Use `\( expression \)` format.
- Block math: Use `\[ expression \]` format.
- **Never use dollar signs** (`$` or `$$`) for LaTeX formatting.

**Example:**

```markdown
The Euclidean distance is calculated as \( d = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2} \).
```

#### 5.1.5 Lists and Tables

- Use unordered lists (bullet points) for non-sequential items.
- Use ordered lists only when sequence or rank matters.
- Never mix ordered and unordered lists within the same section.
- Never nest lists; keep all lists flat.
- Use Markdown tables for comparisons (A vs. B scenarios) instead of bulleted lists.

#### 5.1.6 Cross-References and Citations

- All factual claims must cite their source.
- Citations follow the format `[source_id]` immediately after the statement, with no space.
- Multiple citations per statement are acceptable: `statement[1][2][3]`.
- Never create a bibliography or references section; citations are inline only.

### 5.2 Mathematical Content

#### 5.2.1 Equation Formatting

- All mathematical expressions must use LaTeX notation.
- Inline expressions: `\( f(x) = x^2 + 3x + 2 \)`
- Block-level equations:
  ```
  \[ f(x) = x^2 + 3x + 2 \]
  ```

#### 5.2.2 Derivation and Explanation

- Always provide context before introducing equations.
- Explain variables and notation before using them in equations.
- After equations, provide interpretation and application context.

---

## 6. Telemetry, Logging, and Monitoring

### 6.1 Application Logs

- **Storage Location:** Application logs are stored in the `applogs` directory.
- **Organization:** Logs are organized into date-based directories.
- **Naming Convention:** Use ISO 8601 date format (YYYY-MM-DD) for log directory names.

### 6.2 Logging Standards

- All ROS2 nodes must implement logging via `rclpy.logging`.
- Log levels must be used appropriately:
    - `DEBUG`: Detailed information for diagnosing problems.
    - `INFO`: General informational messages.
    - `WARN`: Warning messages for potentially problematic conditions.
    - `ERROR`: Error messages for serious problems.
    - `FATAL`: Critical errors that prevent node operation.

---

## 7. Collaboration, Version Control, and CI/CD

### 7.1 GitLab Workflows

- **Issue Tracking:** Use GitLab issue workflows as defined in project templates.
- **Merge Requests:** Use default GitLab MR workflows and templates.
- **Optional Enhancements:** GitLab CI/CD pipelines, SAST (Static Application Security Testing), and deployment options are available and optional.

### 7.2 Git Commit Messages

- Write clear, concise commit messages that describe the actual change.
- Use imperative mood: "Add feature" instead of "Added feature" or "Adds feature".
- Reference issue numbers when applicable: "Closes #123".

### 7.3 Rule Updates and Corrections

- When user feedback corrects assistant behavior, propose adding a new explicit rule to this configuration file.
- This ensures the correction is codified and prevents future deviations.
- Rule updates become part of the permanent project standard.

---

## 8. Agent Behavior and Constraints

### 8.1 Factual Accuracy Mandate

- The documentation agent must never fabricate, infer beyond sources, or present speculative content as fact.
- All technical explanations must be traceable to authoritative sources.
- When information is unavailable, state this explicitly rather than guessing.

### 8.2 Source Fidelity

- Paraphrasing is permitted only when it preserves exact technical meaning.
- Direct quotes must use quotation marks and cite the source immediately.
- Synthetic examples are acceptable only if clearly labeled as examples or hypothetical scenarios.

### 8.3 No Speculative Extensions

- Do not extend or invent new applications of existing techniques.
- Do not propose novel methodologies or frameworks.
- All examples and applications must reference established practices or research.

### 8.4 Tone and Language

- Maintain professional, technical tone throughout.
- Use active voice only when the agent performing the action is relevant.
- Prefer passive voice in technical descriptions: "The output is calculated" rather than "The system calculates the output" when the system is implied.
- Avoid hedging language unless uncertainty is inherent to the topic.

### 8.5 Error Handling and Fallback

- If a required piece of information cannot be sourced, document this gap clearly.
- Suggest where such information might be found (e.g., specific research papers, documentation, or standards).
- Do not proceed with unverified information to complete a section.

---

## 9. Quality Assurance and Review

### 9.1 Pre-Submission Checklist

Before any content is considered complete:

- [ ] All factual claims have explicit source citations.
- [ ] No synthetic data is presented as factual.
- [ ] Code examples follow the language-specific style guidelines in this file.
- [ ] All imports are at the top of files (for Python).
- [ ] All functions and classes have docstrings (for Python).
- [ ] Type hints are present on all functions (for Python).
- [ ] No meta-comments or change-description comments exist.
- [ ] Mathematical notation uses LaTeX format (`\( \)` or `\[ \]`).
- [ ] Markdown structure uses proper header hierarchy.
- [ ] No unused imports or dead code.

### 9.2 Consistency Verification

- Ensure consistent terminology across the handbook.
- Verify that related sections reference each other appropriately.
- Check that code examples compile and execute correctly (where applicable).

---

## 10. Appendix: Quick Reference

### 10.1 Forbidden Patterns

| Pattern | Category | Replacement |
|---------|----------|-------------|
| Meta-comments ("Added feature", "Fixed bug") | Comments | Purpose-based comments only |
| Inline imports in functions | Python imports | Top-level imports |
| Magic numbers and strings | Python constants | Named UPPER_SNAKE_CASE constants |
| Dollar signs for LaTeX | Math notation | `\( inline \)` or `\[ block \]` |
| Synthetic data as fact | Data integrity | Sourced data only |
| Speculative inference | Source fidelity | Exact source details only |

### 10.2 Required Formatters by Language

| Language | Formatter | Config File |
|----------|-----------|-------------|
| Python | black | (via pyproject.toml or .black) |
| C/C++ | clang-format | `.clang-format` (config folder) |
| Rust | rustfmt | (built-in to cargo fmt) |
| Bash | shellcheck | (linter, optional) |
| Markdown | (GitHub-flavored standard) | (no config needed) |

### 10.3 Critical Utilities and Dependencies

| Utility | Purpose | Module |
|---------|---------|--------|
| DEFAULT_QOS | ROS2 quality of service | tempoutils.util |
| Time functions | Conversions and arithmetic | tempoutils.time |
| DateTime utilities | UTC handling, timezone conversion | tempoutils.datetime_utils |
| Math helpers | Mathematical operations | tempoutils.math |

---

## Document Information

- **Version:** 1.0
- **Last Updated:** 2025-12-19
- **Status:** Active
- **Maintainer:** Robotics Handbook Project Team
- **Review Cycle:** Quarterly or upon rule corrections

---

End of Instruction File
