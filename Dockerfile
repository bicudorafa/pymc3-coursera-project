ARG BASE_CONTAINER=jupyter/minimal-notebook
FROM $BASE_CONTAINER

LABEL author="bicudorafa"

USER root

RUN conda install -c conda-forge pymc3 theano-pymc mkl mkl-service sunode=0.2.1 plotly
# Switch back to jovyan to avoid accidental container runs as root
USER $NB_UID

# command that starts up the notebook at the end of the dockerfile
CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]