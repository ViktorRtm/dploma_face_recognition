FROM ubuntu:18.04
RUN cd /tmp
RUN git clone --branch v0.7.0 https://github.com/pgvector/pgvector.git
RUN cd pgvector
RUN make
RUN make install