FROM pytorch/torchserve:0.3.0-cpu

COPY src/models/visual_transformer_model.py /home/model-server/
COPY model_store/deployable_model.pt /home/model-server/
COPY src/models/model_handler.py /home/model-server/

USER root
RUN printf "\nservice_envelope=json" >> /home/model-server/config.properties
USER model-server

RUN torch-model-archiver \
  --model-name=visual_transformer \
  --version=1.0 \
  --model-file=/home/model-server/visual_transformer_model.py \
  --serialized-file=/home/model-server/deployable_model.pt \
  --handler=/home/model-server/model_handler:model_handler \
  --export-path=/home/model-server/model-store

CMD ["torchserve", \
     "--start", \
     "--ts-config=/home/model-server/config.properties", \
     "--models", \
     "visual_transformer=visual_transformer.mar"]