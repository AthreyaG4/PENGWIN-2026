FROM pytorch/pytorch:2.12.1-cuda12.6-cudnn9-runtime

ENV PYTHONUNBUFFERED=1

RUN groupadd -r user && useradd -m --no-log-init -r -g user user

RUN mkdir -p \
        /input/images/peripelvic-fracture-ct \
        /output/images/peripelvic-fracture-ct-segmentation \
        /opt/ml/model \
    && chown -R user:user /input /output /opt/ml

USER user

WORKDIR /opt/app

ENV PATH="/home/user/.local/bin:${PATH}"

RUN python -m pip install --user --break-system-packages -U pip

COPY --chown=user:user requirements.txt /opt/app/
RUN python -m pip install --user --break-system-packages -r requirements.txt

COPY --chown=user:user script.py /opt/app/

ENTRYPOINT ["python", "script.py"]
