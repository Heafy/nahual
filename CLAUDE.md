# AGENTS Guidelines for This Repository

This repository contains a Python application. When working on the project interactively
with an agent please follow the guidelines below so that the development experience
continues to work smoothly.

This application purpose is to create a machine learning model that can detect 
custom gestures, this custom gestures is LSM (Lengua de Señas Mexicana) a sign
language for México, the initial scope is to learn every gesture of the alphabet 
in a local machine learning model.

# Keep Dependencies in Sync

* If you add or update dependencies remember to run `uv lock` to update the lockfile

# Reasoning Process

* Always reason step-by-step
* Validate feasibility before proposing scaling solutions
* Explain trade-offs explicitly
* Provide rationale for architectural choices

## Coding Conventions

* Use PEP 8 – Style Guide for Python Code for coding conventions
* Comment every function with her purpose, arguments and a quick explanation of the function
* Do not abbreviate variables, use full name for better readability

## Commit instructions

* Using conventional commits for messages (`<type>` fix:, feat:, build:, chore:, 
ci:, docs:, style:, refactor:, perf:, test:), and later the commit message 
* Always run `black <file_changed>` and `isort <file_changed>` before commiting.

