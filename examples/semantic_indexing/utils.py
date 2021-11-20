# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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

from paddlenlp.ops.faster_transformer.transformer.decoding import transfer_param

def convert_fp16(model, for_paddle=False):
    #print(model)
    model.ptm.pooler.dense.weight = transfer_param(model.ptm.pooler.dense.weight, restore_data=True)
    model.ptm.pooler.dense.bias = transfer_param(model.ptm.pooler.dense.bias, restore_data=True)
    model.emb_reduce_linear.weight = transfer_param(model.emb_reduce_linear.weight, restore_data=True)
    model.emb_reduce_linear.bias = transfer_param(model.emb_reduce_linear.bias, restore_data=True)
    encoder_layers = model.ptm.encoder.layers
    #print("encoder layers before convert:{}".format(encoder_layers))
    for mod in encoder_layers:
        if not for_paddle:
            mod.norm1.weight = transfer_param(mod.norm1.weight, restore_data=True)
            mod.norm1.bias = transfer_param(mod.norm1.bias, is_bias=True, restore_data=True)
            mod.norm2.weight = transfer_param(mod.norm2.weight, restore_data=True)
            mod.norm2.bias = transfer_param(mod.norm2.bias, is_bias=True, restore_data=True)

        mod.linear1.weight = transfer_param(mod.linear1.weight, restore_data=True)
        mod.linear1.bias = transfer_param(mod.linear1.bias, is_bias=True, restore_data=True)

        #print("mod.self_attn.q_proj.weight before convert:{}".format(mod.self_attn.q_proj.weight))
        mod.self_attn.q_proj.weight = transfer_param(
        mod.self_attn.q_proj.weight, restore_data=True)
        #print("mod.self_attn.q_proj.weight after convert:{}".format(mod.self_attn.q_proj.weight))
        mod.self_attn.q_proj.bias = transfer_param(
        mod.self_attn.q_proj.bias, is_bias=True, restore_data=True)
        mod.self_attn.k_proj.weight = transfer_param(
        mod.self_attn.k_proj.weight, restore_data=True)
        mod.self_attn.k_proj.bias = transfer_param(
        mod.self_attn.k_proj.bias, is_bias=True, restore_data=True)
        mod.self_attn.v_proj.weight = transfer_param(
        mod.self_attn.v_proj.weight, restore_data=True)
        mod.self_attn.v_proj.bias = transfer_param(
        mod.self_attn.v_proj.bias, is_bias=True, restore_data=True)
        mod.self_attn.out_proj.weight = transfer_param(
        mod.self_attn.out_proj.weight, restore_data=True)
        mod.self_attn.out_proj.bias = transfer_param(
        mod.self_attn.out_proj.bias, is_bias=True, restore_data=True)

        mod.linear2.weight = transfer_param(mod.linear2.weight, restore_data=True)
        mod.linear2.bias = transfer_param(mod.linear2.bias, is_bias=True, restore_data=True)
    
    encoder_layers = model.ptm.encoder.layers
    #print("encoder layers after convert:{}".format(encoder_layers))
