# Contribute to FEDn

We try to make contributing to our products as easy and transparent as possible and we appreciate your input, whether it's:

- Reporting a bug
- Proposing a feature
- Submitting a fix
- Discussing the current state

## We develop with GitHub

We use github to host our codebase, to track issues and feature requests, as well as accept and merge pull requests. 

We actively welcome your pull requests which, we believe, are the best way to propose changes to the codebase! 

The branching strategy we follow is [GitFlow](https://datasift.github.io/gitflow/IntroducingGitFlow.html#:~:text=GitFlow%20is%20a%20branching%20model,and%20scaling%20the%20development%20team.) and we actively follow the guidelines that come with it.

### GitHub Issues

Report a bug or propose a feature by [opening a new GitHub Issue](https://github.com/scaleoutsystems/scaleout-client/issues/new/choose). We have set up templates accordingly so, choose the one that matches your case. Please try to provide all the required information in the template. If you believe some of the sections are not relevant to your case, you can omit them.

### Branches & Pull Requests

- **main** branch has the latest release of the Scaleout client
- if your branch introduces new functionality, name it **feature/[GitHub-Issue-ID]**
- if your branch resolves a bug, name it **issue/[GitHub-Issue-ID]**
- if your branch is a hotfix, name it **hotfix/[GitHub-Issue-ID]**

Open your pull requests against the **main** branch.

### Code checks
We defined GitHub actions that check code quality and formatting against pushed branches and pull requests. We use:

- [ruff](https://docs.astral.sh/ruff/) for linting, formatting and import sorting. See pyproject.toml for more details.

For more information please refer to the code check action: [.github/workflows/code-checks.yaml](.github/workflows/code-checks.yaml).