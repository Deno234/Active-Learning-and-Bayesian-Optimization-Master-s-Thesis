# Security And Privacy Notes

This release was prepared as a cleaned public-facing repository.

The following categories are intentionally excluded:

- personal shell history and local IDE state;
- `.env` files, tokens, SSH keys, and credentials;
- raw scheduler logs and dashboard remote-state downloads;
- generated archives and temporary folders;
- raw molecular-dynamics trajectories and most checkpoint caches.

The only trained checkpoints intentionally retained are:

```text
models/phase4_ap_sp_fixed_split_ensemble/
models/phase5_initial_replay_point_000/
models/phase3_round001_pre_proposal/
```

They are research artefacts, not credentials, and are tracked through Git LFS.

Before publishing to GitHub, run:

```bash
python scripts/validate_release.py
```

Then manually inspect any reported path-like strings. Cluster paths and usernames
may still appear in historical runbook examples or provenance records where they
are scientifically useful. Replace or annotate them if your institution requires
full anonymisation.
