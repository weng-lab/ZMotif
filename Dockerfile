FROM continuumio/miniconda3
ADD zmotif.yml /tmp/zmotif.yml
RUN conda env create -f /tmp/zmotif.yml