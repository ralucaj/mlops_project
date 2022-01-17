# Base image
FROM python:3.8-slim
WORKDIR /root

# install python 
RUN apt update && \
apt install --no-install-recommends -y build-essential gcc && \
apt-get install -y wget && \
apt clean && rm -rf /var/lib/apt/lists/*

# Copy the necessary files
COPY requirements.txt /root/requirements.txt
COPY setup.py /root/setup.py
COPY src/ /root/src/
COPY .dvc/ /root/.dvc/
COPY data/processed.dvc /root/data/processed.dvc

# Install requirements
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install --upgrade google-cloud-storage
# RUN pip install dvc
# RUN pip install dvc[gs]

# Installs google cloud sdk, this is mostly for using gsutil to export model.
RUN wget -nv \
    https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz && \
    mkdir /root/tools && \
    tar xvzf google-cloud-sdk.tar.gz -C /root/tools && \
    rm google-cloud-sdk.tar.gz && \
    /root/tools/google-cloud-sdk/install.sh --usage-reporting=false \
        --path-update=false --bash-completion=false \
        --disable-installation-options && \
    rm -rf /root/.config/* && \
    ln -s /root/.config /config && \
    # Remove the backup directory that gcloud creates
    rm -rf /root/tools/google-cloud-sdk/.install/.backup

# Path configuration
ENV PATH $PATH:/root/tools/google-cloud-sdk/bin
# Make sure gsutil will use the default service account
RUN echo '[GoogleCompute]\nservice_account = default' > /etc/boto.cfg

# create data folder?
# pull data from the cloud
# RUN dvc pull
RUN mkdir /root/reports
RUN mkdir /root/reports/figures
RUN mkdir /root/models

# Define the application to run when the image is executed
ENTRYPOINT ["python", "-u", "src/models/train_model.py"]