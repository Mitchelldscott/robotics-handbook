# Research Instructions for Robotics Handbook

**Project:** Robotics Handbook | **Version:** 1.0 | **Created:** 2025-12-26

_These instructions guide the research phase for gathering authoritative sources on robotics topics._

## Objective

Conduct systematic, comprehensive research to identify and document all relevant authoritative sources for a given robotics topic. The output shall be a BibTeX file containing verified, citable sources that meet the project's factual accuracy standards.

## Research Scope

Research applies to topics within the **Supported Domains** established in `development_instructions.md`:

- ROS2, Docker
- Physics (dynamics, kinematics, mechanics)
- Robotics (hardware, actuation, sensing)
- Embedded systems
- Mathematics (linear algebra, calculus, differential equations)
- Deep learning, Machine learning
- Control theory
- Autonomous systems
- Data-driven design
- Reinforcement learning

## Research Process

### 1. Topic Definition and Scoping

Clearly define the research subject with specific scope boundaries. Establish the key research questions that the documentation will address. Identify primary and secondary themes within the topic.

Example: For "Mobile Robot Path Planning," primary themes include graph-based planning (Dijkstra, A*), sampling-based planning (RRT, PRM), and reactive planning.

### 2. Source Identification Strategy

Employ multiple channels to identify authoritative sources:

- **Academic Databases:** IEEE Xplore, arXiv, DBLP, Google Scholar
- **Official Documentation:** Framework manuals (ROS2 documentation, Nav2 documentation), library documentation
- **Textbooks and References:** Peer-reviewed robotics textbooks, control theory references, mathematics references
- **Technical Standards:** IEEE standards, ISO standards relevant to robotics
- **Project Repositories:** Official GitHub repositories with comprehensive documentation and references
- **Conference Proceedings:** Top-tier robotics conferences (ICRA, IROS, RSS, CoRL)
- **Journal Articles:** IEEE Transactions, International Journal of Robotics Research, related domain journals

### 3. Search Methodology

Execute systematic searches using multiple query strategies:

- **Keyword-Based Searches:** Use primary topic keywords and technical terminology
- **Author Searches:** Identify prolific researchers in the domain and survey their publications
- **Citation Chaining:** Follow references from high-quality sources to identify related works
- **Topic-Specific Searches:** Use domain-specific terminology and alternative names (synonyms)

Document the search strategies and queries used. Record the date of searches to establish recency.

### 4. Source Evaluation Criteria

Evaluate each identified source against the following criteria:

**Authoritativeness:**
- Author credentials and institutional affiliation
- Publication venue (peer-reviewed journals, established conferences, official documentation)
- Citation count and impact factor (for academic publications)

**Relevance:**
- Direct relation to the research topic
- Level of technical depth appropriate for the handbook
- Contribution of unique or essential information

**Recency:**
- Publication date relative to the research topic (consider technology evolution)
- Currency of information for practical applications
- Presence of updates or revisions

**Verifiability:**
- Availability of the source for consultation
- Sufficient bibliographic information for retrieval
- Accessibility (preference for open-access or institutionally available sources)

Exclude sources that cannot be verified, lack sufficient documentation, or contradict well-established facts without clear justification.

### 5. Source Verification

Verify each source through:

- **Zotero Verification:** Use Zotero to confirm bibliographic data and assess source metadata
- **Direct Access:** Confirm the source is accessible and contains the stated information
- **Cross-Reference:** Compare claims across multiple sources to identify consensus or disputes
- **Institutional Availability:** Verify accessibility through institutional subscriptions or open-access repositories

Document verification status for each source.

### 6. BibTeX File Generation

Export all verified sources as a structured BibTeX file with the following requirements:

**Format Standards:**
- Valid BibTeX syntax (processable by standard LaTeX tools)
- Consistent field ordering: `title`, `author`, `year`, `doi` (if available), `url`, `journal`/`booktitle`, `publisher`

**Required Fields (per entry type):**
- Article: `author`, `title`, `journal`, `year`, (optional: `volume`, `pages`, `doi`)
- InProceedings: `author`, `title`, `booktitle`, `year`, (optional: `pages`, `doi`)
- Book: `author`, `title`, `publisher`, `year`
- Misc: `author`/`organization`, `title`, `year`, `url`, `note`
- Online Resources: `author`/`organization`, `title`, `year`, `url`, `note` (with access date)

**Additional Requirements:**
- Each entry shall have a unique, meaningful citation key (e.g., `Smith2021PathPlanning`)
- URLs shall include access dates for online sources: `note = {Accessed: YYYY-MM-DD}`
- DOI numbers shall be included where available in preference to generic URLs
- Type errors or missing critical fields shall be flagged for correction

### 7. Documentation

For each source, record:

- Full bibliographic information
- Direct relevance to the research topic (brief one-sentence summary)
- Classification (theoretical foundation, practical application, survey, reference implementation)
- Verification status (confirmed, pending, unavailable)
- Access method (institutional subscription, open-access, fee-required)

### 8. Quality Assurance

Before finalizing the BibTeX file:

- Verify all entries have complete, required fields
- Confirm all citations can be resolved (DOI lookup, URL validation)
- Check for duplicate entries and consolidate as appropriate
- Validate BibTeX syntax using automated tools
- Ensure citation keys follow consistent naming conventions

### 9. Scope Documentation

In the BibTeX file header or associated metadata, document:

- Research date(s)
- Search strategies employed
- Total sources identified vs. verified
- Any gaps or limitations in source availability
- Date of last update or verification

## Output Specification

**File Format:** BibTeX (`.bib`)

**Naming Convention:** `{topic}_sources.bib` (e.g., `path_planning_sources.bib`)

**Minimum Coverage:** At least 15–20 verified sources per research topic, organized by:
- Foundational theory (textbooks, seminal papers)
- Current implementations (recent journal articles, conference papers)
- Practical references (documentation, implementation guides)
- Related domains (supporting theory from adjacent fields)

## Integration with Development Workflow

The research phase produces the BibTeX file used in the next phase (`extraction_instructions.md`). The extraction phase will:

1. Use verified sources from the BibTeX file
2. Extract key concepts and technical details
3. Synthesize information into markdown documentation
4. Maintain citation traceability to source materials

## Compliance Notes

Research shall adhere to the **Core Principles** in `development_instructions.md`:

- **Factual Accuracy:** All sources must be verifiable and authoritative
- **No Original Research:** Focus on existing knowledge, established frameworks, proven methodologies
- **Professional Standards:** Document search methodology transparently

Research must respect intellectual property rights and licensing restrictions of identified sources.

---

**Status:** Active | **Last Updated:** 2025-12-26 | **Related Files:** `development_instructions.md`, `extraction_instructions.md`