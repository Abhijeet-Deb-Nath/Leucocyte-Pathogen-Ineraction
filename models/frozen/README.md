# Frozen Models

This directory holds paper-track checkpoints that have been frozen as named reference artifacts.

Current intended main checkpoint:

- `hier_dagger_main.pt`

Meaning:

- hierarchical recurrent host controller
- trained with behavior cloning + DAgger
- main learned controller for GUI and paper experiments

Artifact policy:

- keep large binary checkpoints out of Git when possible
- store a local copy here for GUI/demo use
- keep a backup copy in Google Drive

Expected file placement:

- `models/frozen/hier_dagger_main.pt`

Do not treat experimental smoke checkpoints as frozen models.
