# -*- coding: utf-8 -*-

"""
@Author: Shaoweihua.Liu
@Contact: liushaoweihua@126.com
@Site: github.com/liushaoweihua
@File: train.py
@Time: 2020/3/3 10:37 AM
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import time
import json
import keras
import codecs
import pickle
import numpy as np
from .utils.processor import Processor
from .utils.models import NerCnnModel, NerRnnModel
from .utils.callbacks import NerCallbacks
from .utils.metrics import CrfAcc, CrfLoss


def train(train_data,dev_data,save_path,bert_config,bert_checkpoint,bert_vocab,do_eval=1,device_map="0",tag_padding="X",best_fit=1,max_epochs=10,early_stop_patience=3, \
reduce_lr_patience=3,reduce_lr_factor=0.5,batch_size=64,max_len=64,learning_rate=5e-6,model_type='cnn',cnn_filters=128,cnn_kernel_size=3,cnn_blocks=4,dropout_rate=0,learning_rate=5e-5):
    """模型训练流程
    """
    # 环境设置
    os.environ["CUDA_VISIBLE_DEVICES"] = device_map if device_map != "cpu" else ""
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # 数据准备
    processor = Processor(train_data, bert_vocab, tag_padding)
    train_tokens, train_segs, train_tags = processor.process(train_data, max_len)
    train_x = [np.array(train_tokens), np.array(train_segs)]
    train_y = np.array(train_tags)
    if do_eval:
        dev_tokens, dev_segs, dev_tags = processor.process(dev_data, max_len)
        devs = [[np.array(dev_tokens), np.array(dev_segs)], np.array(dev_tags)]
    else:
        devs = None
    # 模型准备
    if model_type == "cnn":
        ner_model = NerCnnModel(
            bert_config=bert_config,
            bert_checkpoint=bert_checkpoint,
            albert=albert,
            max_len=max_len,
            numb_tags=processor.numb_tags,
            dropout_rate=dropout_rate,
            filters=cnn_filters,
            kernel_size=cnn_kernel_size,
            blocks=cnn_blocks).build()
    elif model_type == "rnn":
        ner_model = NerRnnModel(
            bert_config=bert_config,
            bert_checkpoint=bert_checkpoint,
            albert=albert,
            max_len=max_len,
            numb_tags=processor.numb_tags,
            dropout_rate=dropout_rate,
            cell_type=cell_type,
            units=rnn_units,
            num_hidden_layers=rnn_num_hidden_layers).build()
    else:
        raise ValueError("model_type should be 'cnn' or 'rnn'.")
    crf_accuracy = CrfAcc(processor.tag_to_id, ag_padding).crf_accuracy
    crf_loss = CrfLoss(processor.tag_to_id, tag_padding).crf_loss
    ner_model.compile(
        optimizer=keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-8),
        loss=crf_loss,
        metrics=[crf_accuracy])
    # 模型训练
    bert_type = "ALBERT" if albert else "BERT"
    model_type = "IDCNN-CRF" if model_type == "cnn" else ("BILSTM-CRF" if cell_type == "lstm" else "BIGRU-CRF")
    model_save_path = os.path.abspath(
        os.path.join(save_path, "%s-%s.h5" % (bert_type, model_type)))
    log_save_path = os.path.abspath(
        os.path.join(save_path, "%s-%s-%s.log" % (bert_type, model_type, time.strftime("%Y%m%d_%H%M%S"))))
    if best_fit:
        best_fit_params = {
            "early_stop_patience": early_stop_patience,
            "reduce_lr_patience": reduce_lr_patience,
            "reduce_lr_factor": reduce_lr_factor,
            "save_path": model_save_path
        }
        callbacks = NerCallbacks(processor.id_to_tag, best_fit_params, tag_padding, log_save_path)
        epochs = max_epochs
    else:
        callbacks = NerCallbacks(processor.id_to_tag, None, tag_padding, log_save_path)
        epochs = hard_epochs

    ner_model.fit(
        x=train_x,
        y=train_y,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=devs)

    # 保存信息
    with codecs.open(os.path.join(save_path, "tag_to_id.pkl"), "wb") as f:
        pickle.dump(processor.tag_to_id, f)
    with codecs.open(os.path.join(save_path, "id_to_tag.pkl"), "wb") as f:
        pickle.dump(processor.id_to_tag, f)
    model_configs = {
        "max_len": max_len,
        "tag_padding": tag_padding,
        "model_path": model_save_path,
        "bert_vocab": os.path.abspath(bert_vocab)}
    with codecs.open(os.path.join(save_path, "model_configs.json"), "w") as f:
        json.dump(model_configs, f, ensure_ascii=False, indent=4)
    if not best_fit:
        ner_model.save(model_save_path)
