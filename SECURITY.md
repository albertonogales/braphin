# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 1.0.x   | ✅ Yes    |

## Reporting a Vulnerability

**Please do not open a public GitHub issue for security vulnerabilities.**

Instead, report it privately:

1. Go to the repository's **Security** tab on GitHub.
2. Click **"Report a vulnerability"** to open a private advisory.
3. Include a description of the issue, steps to reproduce, and potential impact.

You will receive an acknowledgement within **72 hours** and a resolution timeline within **14 days**.

## Scope

BRAPHIN is a data-analysis library. Security-relevant concerns include:

- **Path traversal** — e.g. malicious atlas file paths escaping the expected directory
- **Arbitrary code execution** — e.g. unsafe deserialisation of NPY / JSON files supplied by a third party
- **Dependency vulnerabilities** — known CVEs in nibabel, numpy, scipy, or networkx that affect BRAPHIN's usage

Out of scope: issues that require physical access to the machine, or that only affect users who have already compromised their own environment.

## Patient Data

BRAPHIN itself does not transmit, store, or log neuroimaging data. If you discover a code path that could accidentally write subject data to a log file or a publicly readable location, please report it as a vulnerability.
