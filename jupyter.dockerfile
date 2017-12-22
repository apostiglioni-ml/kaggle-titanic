FROM jupyter/scipy-notebook
RUN conda install -y pandas=0.22.0 graphviz seaborn=0.8.1 scikit-learn=0.19.1
