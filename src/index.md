# Data-driven Design of Autonomous Systems

## Introduction

This book is a consolidation of foundations and algorithms for data-driven
autonomous system design. The material is distilled from key reference texts in
Machine Learning and Robotics.

_The material is **NOT** original and may be missing citations._

---

## Outline

The content is organized into three parts (roman numerals), each part has a number of chapters and each chapter has it's own md file. 

### I. Mathematics for Autonomous Systems

The foundational theory and reasoning that enables modeling, analyzing and
designing autonomous systems.

### II. Algorithms and Applications

The fun part that comes after learning math... using it to solve problems.

### III. Computer Architecture and Bare-Metal Programming

How much of the existing open source robotics stack do you really need? Could
some of the layers designed with human users in mind be removed?

---

> **Contributing**: workflows and LLM instructions
>
> 1. Request a thorough list of sources and references for a subject.
>    - export the sources as `.bibtex` file
> 2. Use Zotero to verify the sources and sync data for sources that can be
>    found.
> 3. Condense sources into a context file for an LLM.
> 4. Give the context file, `docs/development_instructions.md` and a short
>    description to the LLM.
> 5. Verify the entire file, add citation/sources when possible.
> 6. Insert a Pluto notebook at the bottom as an example of the subject.
>    - Do a mini-project using the theory from the subject (ask LLM for help
>      brainstorming)
