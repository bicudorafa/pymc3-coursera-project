ARG BASE_CONTAINER=jupyter/minimal-notebook:hub-2.0.0
FROM $BASE_CONTAINER

LABEL author="bicudorafa"

USER root

COPY environment.yml .
RUN conda env update --file environment.yml

# Switch back to jovyan to avoid accidental container runs as root
USER $NB_UID

# command that starts up the notebook at the end of the dockerfile
CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]