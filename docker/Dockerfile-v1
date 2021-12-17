# base image (host OS) setting
FROM jupyter/minimal-notebook
RUN mkdir repo
WORKDIR /repo
COPY . .
# main dependencies installation and personal modules testing
#RUN pip install --upgrade pip setuptools wheel && pip install -r requirements.txt
RUN conda install -c conda-forge pymc3 theano-pymc mkl mkl-service sunode=0.2.1
#RUN pip install jupyter
#RUN py.test
# command that starts up the notebook at the end of the dockerfile
CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]