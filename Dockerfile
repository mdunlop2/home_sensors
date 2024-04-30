FROM quay.io/jupyter/scipy-notebook:2024-04-22

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Install additional OS dependencies
USER root
RUN apt-get update --yes && \
 apt-get install --yes --no-install-recommends \
 sqlite3
USER ${NB_UID}

# Install pip requirements
COPY requirements.txt .
RUN mamba install --file requirements.txt

ENV PYTHONPATH=/home/$NB_USER/work:$PYTHONPATH
