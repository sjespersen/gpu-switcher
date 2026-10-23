# Working in this repo

- Commit straight to `main` and push; no feature branches or pull requests.
- Only the Python standard library: no dependencies, so the image is the app and a slim Python.
- A release is a tag `vX.Y.Z`: CI publishes `ghcr.io/sjespersen/gpu-switcher`, and
  homelab-agent pins it by digest (its `update-images.sh` moves the pin).
