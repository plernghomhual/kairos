# Contributing

## Branches

Use `agent-NNN-description` for branch names.

Example: `agent-006-hmm-tuning`

## Commits

Use `kairos(module): description`.

Example: `kairos(regime): add HMM hyperparameter optimization`

## Pull Requests

Use `Agent N: Title` for PR titles.

Example: `Agent 6: HMM Hyperparameter Optimization`

Include these sections in every PR body:

```md
## Changes

## Testing

## Risk
```

## Merge Strategy

Use squash-merge with linear history.

Before opening a PR, run:

```sh
make test
make lint
```
