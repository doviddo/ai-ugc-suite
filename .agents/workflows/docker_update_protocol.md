---
description: How to correctly update and deploy python code changes when using Docker on a remote server.
---
# Updating Python Code in Docker Containers

When making changes to Python backend code (like `app.py`) in a containerized environment, relying purely on restarting the container WILL NOT WORK if the code is copied into the image rather than mounted as a volume.

## Problem
In our `docker-compose.yml`, the `app` code directory is NOT mounted as a volume. Instead, the `Dockerfile` uses a command like `COPY . .` to bake the Python code into the image at build time.

When downloading changes via `git pull` or manual ZIP extraction on the remote host:
- Running `docker compose restart` only restarts the container with the **old baked-in code**. 
- The newly downloaded `.py` files on the host system are completely ignored.

## Solution
To apply code changes, you MUST trigger a rebuild of the Docker image so the new files get baked in.

**Correct Command:**
```bash
docker compose up -d --build
```

**Incorrect Command (Causes old bugs to persist):**
```bash
docker compose restart
```

## Exception
If you are ONLY editing files located inside a directory that is explicitly mounted as a volume in `docker-compose.yml` (e.g. `./output:/app/output`), those changes are reflected instantly inside the container. Everything else requires `--build`.
