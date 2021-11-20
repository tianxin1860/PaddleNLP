# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import abc
import sys

import numpy as np

import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class SemanticIndexBase(nn.Layer):
    def __init__(self, pretrained_model, dropout=None, output_emb_size=None, use_fp16=False):
        super().__init__()
        self.ptm = pretrained_model
        self.dropout = nn.Dropout(dropout if dropout is not None else 0.1)

        # if output_emb_size is not None, then add Linear layer to reduce embedding_size, 
        # we recommend set output_emb_size = 256 considering the trade-off beteween 
        # recall performance and efficiency

        self.output_emb_size = output_emb_size
        if output_emb_size > 0:
            weight_attr = paddle.ParamAttr(
                initializer=paddle.nn.initializer.TruncatedNormal(std=0.02))
            self.emb_reduce_linear = paddle.nn.Linear(
                768, output_emb_size, weight_attr=weight_attr)

        self.use_fp16 = use_fp16

    def get_pooled_embedding(self,
                             input_ids,
                             token_type_ids=None,
                             position_ids=None,
                             attention_mask=None):
        
        if self.use_fp16:
            if attention_mask is None:
                attention_mask = paddle.unsqueeze(
                (input_ids == self.ptm.pad_token_id
                 ).astype(self.ptm.pooler.dense.weight.dtype) * -1e4,
                axis=[1, 2])
                #print("[DEBUG]attention_mask val in paddlenlp is 1e4")

            embedding_output = self.ptm.embeddings(
                input_ids=input_ids,
                position_ids=position_ids,
                token_type_ids=token_type_ids)

            # print("embedding_output:{}".format(embedding_output))
            # print("attention_mask:{}".format(attention_mask))
            # print(self.ptm.encoder)

            embedding_output = paddle.cast(embedding_output, 'float16')    
            attention_mask = paddle.cast(attention_mask, 'float16')

            encoder_outputs = self.ptm.encoder(embedding_output, attention_mask)
            cls_embedding = self.ptm.pooler(encoder_outputs)
        else:
            _, cls_embedding = self.ptm(input_ids, token_type_ids, position_ids,
                                        attention_mask)

        if self.output_emb_size > 0:
            cls_embedding = self.emb_reduce_linear(cls_embedding)
        cls_embedding = self.dropout(cls_embedding)
        cls_embedding = paddle.cast(cls_embedding, 'float32')
        cls_embedding = F.normalize(cls_embedding, p=2, axis=-1)

        return cls_embedding

    def get_semantic_embedding(self, data_loader):
        self.eval()
        with paddle.no_grad():
            for batch_data in data_loader:
                input_ids, token_type_ids = batch_data
                input_ids = paddle.to_tensor(input_ids)
                token_type_ids = paddle.to_tensor(token_type_ids)

                text_embeddings = self.get_pooled_embedding(
                    input_ids, token_type_ids=token_type_ids)

                yield text_embeddings

    def cosine_sim(self,
                   query_input_ids,
                   title_input_ids,
                   query_token_type_ids=None,
                   query_position_ids=None,
                   query_attention_mask=None,
                   title_token_type_ids=None,
                   title_position_ids=None,
                   title_attention_mask=None):

        query_cls_embedding = self.get_pooled_embedding(
            query_input_ids, query_token_type_ids, query_position_ids,
            query_attention_mask)

        title_cls_embedding = self.get_pooled_embedding(
            title_input_ids, title_token_type_ids, title_position_ids,
            title_attention_mask)

        cosine_sim = paddle.sum(query_cls_embedding * title_cls_embedding,
                                axis=-1)
        return cosine_sim

    @abc.abstractmethod
    def forward(self):
        pass
