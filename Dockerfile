FROM ubuntu:18.04

ENV DEBIAN_FRONTEND="noninteractive" TZ="America/New_York"

RUN apt-get update && apt-get install -y \
    libopenmpi-dev \
    openmpi-bin \
    ghostscript \
    libgs-dev \
    libgd-dev \
    libexpat1-dev \
    zlib1g-dev \
    libxml2-dev \
    autoconf automake libtool \
    libhtml-template-compiled-perl \
    libxml-opml-simplegen-perl \
    libxml-libxml-debugging-perl \
    sudo \
    openssh-server \ 
    python3-pip \
    git-all

RUN PERL_MM_USE_DEFAULT=1 perl -MCPAN -e 'install Log::Log4perl'
RUN PERL_MM_USE_DEFAULT=1 perl -MCPAN -e 'install Math::CDF'
RUN PERL_MM_USE_DEFAULT=1 perl -MCPAN -e 'install CGI'
RUN PERL_MM_USE_DEFAULT=1 perl -MCPAN -e 'install HTML::PullParser'
RUN PERL_MM_USE_DEFAULT=1 perl -MCPAN -e 'install HTML::Template'
RUN PERL_MM_USE_DEFAULT=1 perl -MCPAN -e 'install XML::Simple'
RUN PERL_MM_USE_DEFAULT=1 perl -MCPAN -e 'install XML::Parser::Expat'
RUN PERL_MM_USE_DEFAULT=1 perl -MCPAN -e 'install XML::LibXML'
RUN PERL_MM_USE_DEFAULT=1 perl -MCPAN -e 'install XML::LibXML::Simple'
RUN PERL_MM_USE_DEFAULT=1 perl -MCPAN -e 'install XML::Compile'
RUN PERL_MM_USE_DEFAULT=1 perl -MCPAN -e 'install XML::Compile::SOAP11'
RUN PERL_MM_USE_DEFAULT=1 perl -MCPAN -e 'install XML::Compile::WSDL11'
RUN PERL_MM_USE_DEFAULT=1 perl -MCPAN -e 'install XML::Compile::Transport::SOAPHTTP'

RUN mkdir /opt/meme
ADD http://meme-suite.org/meme-software/5.3.0/meme-5.3.0.tar.gz /opt/meme
WORKDIR /opt/meme/
RUN tar zxvf meme-5.3.0.tar.gz && rm -fv meme-5.3.0.tar.gz
RUN cd /opt/meme/meme-5.3.0 && \
	./configure --prefix=/opt  --enable-build-libxml2 --enable-build-libxslt  --with-url=http://meme-suite.org && \ 
	make && \
	make install && \
        rm -rfv /opt/meme
ENV PATH="/opt/bin:${PATH}"

RUN mkdir /opt/bedtools
WORKDIR /opt/bedtools/
RUN wget https://github.com/arq5x/bedtools2/releases/download/v2.30.0/bedtools.static.binary && \
    mv bedtools.static.binary bedtools && \
    chmod a+x bedtools

ENV PATH="/opt/bedtools:${PATH}"

RUN pip3 install --upgrade pip && pip3 install keras tensorflow==1.15 scikit-learn pyfaidx pybedtools

RUN wget https://github.com/gmarcais/Jellyfish/releases/download/v2.3.0/jellyfish-2.3.0.tar.gz && \
    tar -xzvf jellyfish-2.3.0.tar.gz && \
    cd jellyfish-2.3.0 && ./configure --prefix=/opt && make -j 4 && make install 