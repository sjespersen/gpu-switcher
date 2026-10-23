# Only the standard library: the app and its page on a slim Python. nvidia-smi (the GPU gauge)
# comes from the NVIDIA Container Toolkit when the container gets the GPU (CDI), not from here.
FROM python:3.13-slim@sha256:70729b46c69b4f1e97c4822c1af3df53a1476cf5ddc6c087c0c10bc3a5678c2f
COPY app.py page.html login.html style.css /app/
USER 1000:1000
EXPOSE 8079
CMD ["python", "/app/app.py"]
