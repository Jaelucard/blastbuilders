# The scorer builds this on linux/amd64 and runs it with no network. The
# digest is the image we labeled every commit in: its Python and SQLite
# decide which sqlglot tests run, so keep it if your image runs them.
FROM python:3.13-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SKIP_INTEGRATION=0 \
    PYTHONPATH=/app/sqlglot

WORKDIR /app

COPY requirements.lock /app/requirements.lock
RUN pip install --no-cache-dir -r /app/requirements.lock

RUN python -c "import tarfile, urllib.request, io, os; \
    url = 'https://github.com/tobymao/sqlglot/archive/eb11aec4be8cb2122afe713a4816662c3f0fb2fa.tar.gz'; \
    data = urllib.request.urlopen(url).read(); \
    tf = tarfile.open(fileobj=io.BytesIO(data), mode='r:gz'); \
    tf.extractall('/app'); \
    os.rename('/app/sqlglot-eb11aec4be8cb2122afe713a4816662c3f0fb2fa', '/app/sqlglot')" \
    && python -m compileall -q /app/sqlglot/sqlglot /app/sqlglot/tests \
    && chmod -R a+rX /app/sqlglot

COPY build_tables_cache.py /app/build_tables_cache.py
RUN python /app/build_tables_cache.py /app/maps/tables.pkl

COPY maps/maps.zlib /app/maps/maps.zlib
COPY patch_utils.py packer.py impact.py verifier.py selector.py /app/
RUN python -m compileall -q /app/patch_utils.py /app/packer.py /app/impact.py /app/verifier.py /app/selector.py \
    && chmod -R a+rX /app

CMD ["python", "/app/selector.py"]
