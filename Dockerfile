FROM lmsysorg/sglang@sha256:fce5585fa8da175224ead70727a4660f719e34adf0638aebf6dd27930f03e1b6
COPY requirements.lock /tmp/reference-requirements.lock
RUN python3 -m pip install --no-cache-dir --no-deps -r /tmp/reference-requirements.lock
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
ENTRYPOINT ["python3", "-m", "server.supervisor"]
