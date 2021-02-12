import numbers,math
from copy import copy

import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

from collections import OrderedDict



def extract_top_level_dict(current_dict):
    """
    Builds a graph dictionary from the passed depth_keys, value pair. Useful for dynamically passing external params
    :param depth_keys: A list of strings making up the name of a variable. Used to make a graph for that params tree.
    :param value: Param value
    :param key_exists: If none then assume new dict, else load existing dict and add new key->value pairs to it.
    :return: A dictionary graph of the params already added to the graph.
    """
    output_dict = dict()
    for key in current_dict.keys():
        name = key.replace("layer_dict.", "")
        name = name.replace("layer_dict.", "")
        name = name.replace("block_dict.", "")
        name = name.replace("module-", "")
        top_level = name.split(".")[0]
        try:
            sub_level = ".".join(name.split(".")[1:])
        except IndexError:
            return current_dict

        if top_level not in output_dict:
            if sub_level == "":
                output_dict[top_level] = current_dict[key]
            else:
                output_dict[top_level] = {sub_level: current_dict[key]}
        else:
            new_item = {key: value for key, value in output_dict[top_level].items()}
            new_item[sub_level] = current_dict[key]
            output_dict[top_level] = new_item

    return output_dict


class MetaEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, padding_idx):
        super(MetaEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        if padding_idx is not None:
            if padding_idx > 0:
                assert padding_idx < self.num_embeddings, 'Padding_idx must be within num_embeddings'
            elif padding_idx < 0:
                assert padding_idx >= -self.num_embeddings, 'Padding_idx must be within num_embeddings'
                padding_idx = self.num_embeddings + padding_idx

        self.padding_idx = padding_idx
        self.weight = nn.Parameter(torch.Tensor(num_embeddings, embedding_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight)
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)

    def forward(self, x, params=None):
        if params is not None:
            params = extract_top_level_dict(current_dict=params)
            (weight) = params["weight"]
            assert weight.shape == self.weight.shape
        else:
            weight = self.weight

        return F.embedding(x, weight, self.padding_idx)


class MetaConv1dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, use_bias=True, groups=1, dilation_rate=1):
        """
        A MetaConv1D layer. Applies the same functionality of a standard Conv1D layer with the added functionality of
        being able to receive a parameter dictionary at the forward pass which allows the convolution to use external
        weights instead of the internal ones stored in the conv layer. Useful for inner loop optimization in the meta
        learning setting.
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param kernel_size: Convolutional kernel size
        :param stride: Convolutional stride
        :param padding: Convolution padding
        :param use_bias: Boolean indicating whether to use a bias or not.
        """
        super(MetaConv1dLayer, self).__init__()
        self.stride = int(stride)
        self.padding = int(padding)
        self.dilation_rate = int(dilation_rate)
        self.use_bias = use_bias
        self.groups = int(groups)
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size))
        nn.init.xavier_uniform_(self.weight)

        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x, params=None):
        """
        Applies a conv1D forward pass. If params are not None will use the passed params as the conv weights and biases
        :param x: Input image batch.
        :param params: If none, then conv layer will use the stored self.weights and self.bias, if they are not none
        then the conv layer will use the passed params as its parameters.
        :return: The output of a convolutional function.
        """
        if params is not None:
            params = extract_top_level_dict(current_dict=params)
            if self.use_bias:
                (weight, bias) = params["weight"], params["bias"]
            else:
                (weight) = params["weight"]
                bias = None
            assert weight.shape == self.weight.shape
            assert bias.shape == self.bias.shape
        else:
            #print("No inner loop params")
            if self.use_bias:
                weight, bias = self.weight, self.bias
            else:
                weight = self.weight
                bias = None

        out = F.conv1d(input=x, weight=weight, bias=bias, stride=self.stride,
                       padding=self.padding, dilation=self.dilation_rate, groups=self.groups)
        return out


class MetaLinearLayer(nn.Module):
    def __init__(self, input_shape, num_filters, use_bias=True):
        """
        A MetaLinear layer. Applies the same functionality of a standard linearlayer with the added functionality of
        being able to receive a parameter dictionary at the forward pass which allows the convolution to use external
        weights instead of the internal ones stored in the linear layer. Useful for inner loop optimization in the meta
        learning setting.
        :param input_shape: The shape of the input data, in the form (b, f)
        :param num_filters: Number of output filters
        :param use_bias: Whether to use biases or not.
        """
        super(MetaLinearLayer, self).__init__()
        
        try:
            b, c = input_shape
        except TypeError:
            c = input_shape

        self.use_bias = use_bias
        self.weights = nn.Parameter(torch.ones(num_filters, c))
        nn.init.kaiming_uniform(self.weights, a=math.sqrt(5))
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(num_filters))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
       

    def forward(self, x, params=None):
        """
        Forward propagates by applying a linear function (Wx + b). If params are none then internal params are used.
        Otherwise passed params will be used to execute the function.
        :param x: Input data batch, in the form (b, f)
        :param params: A dictionary containing 'weights' and 'bias'. If params are none then internal params are used.
        Otherwise the external are used.
        :return: The result of the linear function.
        """
        if params is not None:
            params = extract_top_level_dict(current_dict=params)
            
            if self.use_bias:
                (weight, bias) = params["weights"], params["bias"]
            else:
                (weight) = params["weights"]
                bias = None
        else:
            pass
            #print('no inner loop params', self)

            if self.use_bias:
                weight, bias = self.weights, self.bias
            else:
                weight = self.weights
                bias = None
        # print(x.shape)
        out = F.linear(input=x, weight=weight, bias=bias)
        return out


class MetaBatchNormLayer(nn.Module):
    def __init__(self, num_features, device, args, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, meta_batch_norm=True, no_learnable_params=False,
                 use_per_step_bn_statistics=False):
        """
        A MetaBatchNorm layer. Applies the same functionality of a standard BatchNorm layer with the added functionality of
        being able to receive a parameter dictionary at the forward pass which allows the convolution to use external
        weights instead of the internal ones stored in the conv layer. Useful for inner loop optimization in the meta
        learning setting. Also has the additional functionality of being able to store per step running stats and per step beta and gamma.
        :param num_features:
        :param device:
        :param args:
        :param eps:
        :param momentum:
        :param affine:
        :param track_running_stats:
        :param meta_batch_norm:
        :param no_learnable_params:
        :param use_per_step_bn_statistics:
        """
        super(MetaBatchNormLayer, self).__init__()
        self.num_features = num_features
        self.eps = eps

        self.affine = affine
        self.track_running_stats = track_running_stats
        self.meta_batch_norm = meta_batch_norm
        self.num_features = num_features
        self.device = device
        self.use_per_step_bn_statistics = use_per_step_bn_statistics
        self.args = args
        self.learnable_gamma = self.args.learnable_bn_gamma
        self.learnable_beta = self.args.learnable_bn_beta

        if use_per_step_bn_statistics:
            self.running_mean = nn.Parameter(torch.zeros(args.number_of_training_steps_per_iter, num_features),
                                             requires_grad=False)
            self.running_var = nn.Parameter(torch.ones(args.number_of_training_steps_per_iter, num_features),
                                            requires_grad=False)
            self.bias = nn.Parameter(torch.zeros(args.number_of_training_steps_per_iter, num_features),
                                     requires_grad=self.learnable_beta)
            self.weight = nn.Parameter(torch.ones(args.number_of_training_steps_per_iter, num_features),
                                       requires_grad=self.learnable_gamma)
        else:
            self.running_mean = nn.Parameter(torch.zeros(num_features), requires_grad=False)
            self.running_var = nn.Parameter(torch.zeros(num_features), requires_grad=False)
            self.bias = nn.Parameter(torch.zeros(num_features),
                                     requires_grad=self.learnable_beta)
            self.weight = nn.Parameter(torch.ones(num_features),
                                       requires_grad=self.learnable_gamma)

        if self.args.enable_inner_loop_optimizable_bn_params:
            self.bias = nn.Parameter(torch.zeros(num_features),
                                     requires_grad=self.learnable_beta)
            self.weight = nn.Parameter(torch.ones(num_features),
                                       requires_grad=self.learnable_gamma)

        self.backup_running_mean = torch.zeros(self.running_mean.shape)
        self.backup_running_var = torch.ones(self.running_var.shape)

        self.momentum = momentum

    def forward(self, input, num_step, params=None, training=False, backup_running_statistics=False):
        """
        Forward propagates by applying a bach norm function. If params are none then internal params are used.
        Otherwise passed params will be used to execute the function.
        :param input: input data batch, size either can be any.
        :param num_step: The current inner loop step being taken. This is used when we are learning per step params and
         collecting per step batch statistics. It indexes the correct object to use for the current time-step
        :param params: A dictionary containing 'weight' and 'bias'.
        :param training: Whether this is currently the training or evaluation phase.
        :param backup_running_statistics: Whether to backup the running statistics. This is used
        at evaluation time, when after the pass is complete we want to throw away the collected validation stats.
        :return: The result of the batch norm operation.
        """
        if params is not None:
            params = extract_top_level_dict(current_dict=params)
            (weight, bias) = params["weight"], params["bias"]
            #print(num_step, params['weight'])
        else:
            #print(num_step, "no params")
            weight, bias = self.weight, self.bias

        if self.use_per_step_bn_statistics:
            running_mean = self.running_mean[num_step]
            running_var = self.running_var[num_step]
            if params is None:
                if not self.args.enable_inner_loop_optimizable_bn_params:
                    bias = self.bias[num_step]
                    weight = self.weight[num_step]
        else:
            running_mean = None
            running_var = None


        if backup_running_statistics and self.use_per_step_bn_statistics:
            self.backup_running_mean.data = copy(self.running_mean.data)
            self.backup_running_var.data = copy(self.running_var.data)

        momentum = self.momentum

        output = F.batch_norm(input, running_mean, running_var, weight, bias,
                              training=True, momentum=momentum, eps=self.eps)

        return output

    def restore_backup_stats(self):
        """
        Resets batch statistics to their backup values which are collected after each forward pass.
        """
        if self.use_per_step_bn_statistics:
            self.running_mean = nn.Parameter(self.backup_running_mean.to(device=self.device), requires_grad=False)
            self.running_var = nn.Parameter(self.backup_running_var.to(device=self.device), requires_grad=False)

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)


class DeepDTAv2Meta(nn.Module):
        
    def __init__(self, args, device):     
        super(DeepDTAv2Meta, self).__init__()
       
        self.args = args
        self.device = device
        self.ligand_embedding = MetaEmbedding(65, 128, padding_idx=0)
        self.protein_embedding = MetaEmbedding(26, 128, padding_idx=0)

        self.dense1 = MetaLinearLayer(32*6, 1024)
        self.dense2 = MetaLinearLayer(1024, 1024)
        self.dense3 = MetaLinearLayer(1024, 512)
        self.out_layer = MetaLinearLayer(512, 1)
        
        self.lig_norm1 = MetaBatchNormLayer(32, self.device, self.args)
        self.lig_norm2 = MetaBatchNormLayer(32*2, self.device, self.args)
        self.lig_norm3 = MetaBatchNormLayer(32*3, self.device, self.args)

        self.prot_norm1 = MetaBatchNormLayer(32, self.device, self.args)
        self.prot_norm2 = MetaBatchNormLayer(32*2, self.device, self.args)
        self.prot_norm3 = MetaBatchNormLayer(32*3, self.device, self.args)

        self.lig_conv1 =  MetaConv1dLayer(128, 32, 6)
        self.lig_conv2 =  MetaConv1dLayer(32, 32*2, 6)
        self.lig_conv3 =  MetaConv1dLayer(32*2, 32*3, 6)

        self.prot_conv1 =  MetaConv1dLayer(128, 32, 8)
        self.prot_conv2 =  MetaConv1dLayer(32, 32*2, 8)
        self.prot_conv3 =  MetaConv1dLayer(32*2, 32*3, 8)


    def forward(self, smiles, protein, num_step, params=None, training=False, \
                                                    backup_running_statistics=False):

        keys = ["ligand_embedding", "protein_embedding", "lig_conv1", "lig_conv2",
                "lig_conv3", "lig_norm1", "lig_norm2", "lig_norm3", "prot_conv1", "prot_conv2",
                "prot_conv3", "prot_norm1", "prot_norm2", "prot_norm3", "dense1",
                "dense2", "dense3", "out_layer"]

        if params is not None:
            params = {key: value[0] for key, value in params.items()}
            params = extract_top_level_dict(params)

        for key in keys:
            if not(key in params):
                params[key] = None

        encode_smiles = self.ligand_embedding(smiles, params=params["ligand_embedding"])\
                                                                        .transpose(2,1)
        encode_protein = self.protein_embedding(protein, params=params["protein_embedding"])\
                                                                        .transpose(2,1)
        

        encode_smiles = self.lig_conv1(encode_smiles, params=params["lig_conv1"])
        encode_smiles = self.lig_norm1(encode_smiles, num_step, params=params["lig_norm1"],\
                training=training, backup_running_statistics=backup_running_statistics)
        encode_smiles = F.dropout(F.leaky_relu(encode_smiles), 0.1)
        encode_smiles = self.lig_conv2(encode_smiles, params=params["lig_conv2"])
        encode_smiles = self.lig_norm2(encode_smiles, num_step, params=params["lig_norm2"],\
                training=training, backup_running_statistics=backup_running_statistics)
        encode_smiles = F.dropout(F.leaky_relu(encode_smiles), 0.1)
        encode_smiles = self.lig_conv3(encode_smiles, params=params["lig_conv3"])
        encode_smiles = self.lig_norm3(encode_smiles, num_step, params=params["lig_norm3"],\
                training=training, backup_running_statistics=backup_running_statistics)
        encode_smiles = F.dropout(F.leaky_relu(encode_smiles), 0.1)

        
        encode_protein = self.prot_conv1(encode_protein, params=params["prot_conv1"])
        encode_protein = self.prot_norm1(encode_protein, num_step, params=params["prot_norm1"],\
                training=training, backup_running_statistics=backup_running_statistics)
        encode_protein = F.dropout(F.leaky_relu(encode_protein), 0.1)
        encode_protein = self.prot_conv2(encode_protein, params=params["prot_conv2"])
        encode_protein = self.prot_norm2(encode_protein, num_step, params=params["prot_norm2"],\
                training=training, backup_running_statistics=backup_running_statistics)
        encode_protein = F.dropout(F.leaky_relu(encode_protein), 0.1)
        encode_protein = self.prot_conv3(encode_protein, params=params["prot_conv3"])
        encode_protein = self.prot_norm3(encode_protein, num_step, params=params["prot_norm3"],\
                training=training, backup_running_statistics=backup_running_statistics)
        encode_protein = F.dropout(F.leaky_relu(encode_protein), 0.1)
    
        ligand_arm_out = encode_smiles
        protein_arm_out = encode_protein

        encode_smiles, _ = torch.max(ligand_arm_out, 2)
        encode_protein, _ = torch.max(protein_arm_out, 2)     
        encode_interaction = torch.cat([encode_smiles, encode_protein], 1)
        
        batch, length = encode_interaction.shape
        encode_interaction = encode_interaction.view(batch, 1, length)
        out = F.dropout(F.leaky_relu(self.dense1(encode_interaction, params=params["dense1"])), 0.1)
        out = F.dropout(F.leaky_relu(self.dense2(out, params=params["dense2"])), 0.1)
        out = F.dropout(F.leaky_relu(self.dense3(out, params=params["dense3"])), 0.1)
        out = self.out_layer(out, params=params["out_layer"]).view(batch, 1)
        return out

    
    def zero_grad(self, params=None):
        if params is None:
            for param in self.parameters():
                if param.requires_grad == True:
                    if param.grad is not None:
                        if torch.sum(param.grad) > 0:
                            print(param.grad)
                            param.grad.zero_()
        else:
            for name, param in params.items():
                if param.requires_grad == True:
                    if param.grad is not None:
                        if torch.sum(param.grad) > 0:
                            print(param.grad)
                            param.grad.zero_()
                            params[name].grad = None

    def restore_backup_stats(self):
        """
        Reset stored batch statistics from the stored backup.
        """
        self.lig_norm1.restore_backup_stats()
        self.lig_norm2.restore_backup_stats()
        self.lig_norm3.restore_backup_stats()
        self.prot_norm1.restore_backup_stats()
        self.prot_norm2.restore_backup_stats()
        self.prot_norm3.restore_backup_stats()
    

    def load_from_path(self, path):
        pretrained_dict = torch.load(path)
        new_dict = self.convert_deepdta_weight(pretrained_dict)
        self.load_state_dict(new_dict)

    def convert_deepdta_weight(self, pretrained_dict):
        print("PRETRAINED", pretrained_dict["ligand_arm.0.weight"].mean())
        new_dict = OrderedDict()
        for k, v in pretrained_dict.items():
            if not ("arm" in k):
                new_k = k
                if not ("embedding" in k):
                    new_k = k.replace("weight", "weights")
            else:
                layer_num = int(k.split(".")[1])

                if "num_batches_tracked" in k:
                    continue

                if "ligand" in k:
                    if layer_num == 0:
                        new_k = k.replace("ligand_arm.0" ,"lig_conv1")
                    elif layer_num == 4:
                        new_k = k.replace("ligand_arm.4" ,"lig_conv2")
                    elif layer_num == 8:
                        new_k = k.replace("ligand_arm.8" ,"lig_conv3")
                    elif layer_num == 1:
                        new_k = k.replace("ligand_arm.1" ,"lig_norm1")
                    elif layer_num == 5:
                        new_k = k.replace("ligand_arm.5" ,"lig_norm2")
                    elif layer_num == 9:
                        new_k = k.replace("ligand_arm.9" ,"lig_norm3")

                elif "protein" in k:
                    if layer_num == 0:
                        new_k = k.replace("protein_arm.0" ,"prot_conv1")
                    elif layer_num == 4:
                        new_k = k.replace("protein_arm.4" ,"prot_conv2")
                    elif layer_num == 8:
                        new_k = k.replace("protein_arm.8" ,"prot_conv3")
                    elif layer_num == 1:
                        new_k = k.replace("protein_arm.1" ,"prot_norm1")
                    elif layer_num == 5:
                        new_k = k.replace("protein_arm.5" ,"prot_norm2")
                    elif layer_num == 9:
                        new_k = k.replace("protein_arm.9" ,"prot_norm3")

            new_dict[new_k] = v
        return new_dict 


if __name__ == "__main__":
    from utils.parser_utils import get_args
    args, device = get_args()
    model = DeepDTAv2Meta(args=args, device="cpu")

    print("Initial", model.state_dict()["lig_conv1.weight"].mean())
    model.load_from_path("DEEPDTAV2_SAMPLE_WEIGHT.pth")
    print("After", model.state_dict()["lig_conv1.weight"].mean())
