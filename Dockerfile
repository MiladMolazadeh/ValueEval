# First stage
FROM python:3.8-slim-buster as builder

COPY . /app
WORKDIR /app
RUN --mount=type=cache,target=/root/.cache pip install --upgrade pip && pip install -r requirements.txt

FROM python:3.8-slim-buster
COPY --from=builder /usr/local/ /usr/local/
COPY --from=builder /app/ /app/
ENV PATH=/usr/local/bin:$PATH
ENV PYTHONPATH="${PYTHONPATH}:/app/plm"
#set workdir
WORKDIR /app/
CMD ["bash"]
