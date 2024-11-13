# [xin] MNSIM2.0 Chiplet
import os
import argparse
import math
import torch
import pickle

import torchvision.models as models
import torch.nn as nn

class Conv_Act_Pool_layer(nn.Module):
    def __init__(self, item):
        super(Conv_Act_Pool_layer, self).__init__()
        
        self.conv_layer = nn.Conv2d(in_channels=item['conv_num_ichan'], out_channels=item['conv_num_ochan'], kernel_size=item['conv_kernel_size'][0], stride=item['conv_strides'][0], padding=item['conv_pads'][0])
        self.bn_layer = nn.BatchNorm2d(num_features=item['conv_num_ochan'], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.activation = nn.ReLU()
        # self.maxpool = nn.MaxPool2d(kernel_size=item['pool_kernel_size'][0], stride=item['pool_strides'][0], padding=item['pool_pads'][0], ceil_mode=False if item['pool_ceil_mode'] == 0 else True)
        self.maxpool = nn.MaxPool2d(kernel_size=item['pool_kernel_size'][0], stride=item['pool_strides'][0])
        
    def forward(self, x):
        x = self.conv_layer(x)
        x = self.bn_layer(x)
        x = self.activation(x)
        x = self.maxpool(x)
        return x
    
class Conv_Act_layer(nn.Module):
    def __init__(self, item):
        super(Conv_Act_layer, self).__init__()
        
        self.conv_layer = nn.Conv2d(in_channels=item['conv_num_ichan'], out_channels=item['conv_num_ochan'], kernel_size=item['conv_kernel_size'][0], stride=item['conv_strides'][0], padding=item['conv_pads'][0])
        self.bn_layer = nn.BatchNorm2d(num_features=item['conv_num_ochan'], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        x = self.conv_layer(x)
        x = self.bn_layer(x)
        x = self.activation(x)
        return x
    
class Conv_Add_Act_layer(nn.Module):
    def __init__(self, item):
        super(Conv_Add_Act_layer, self).__init__()
        
        self.conv_layer = nn.Conv2d(in_channels=item['conv_num_ichan'], out_channels=item['conv_num_ochan'], kernel_size=item['conv_kernel_size'][0], stride=item['conv_strides'][0], padding=item['conv_pads'][0])
        self.bn_layer = nn.BatchNorm2d(num_features=item['conv_num_ochan'], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.addition = nn.quantized.FloatFunctional()
        self.activation = nn.ReLU()
        
    def forward(self, x):
        x = self.conv_layer(x)
        x = self.bn_layer(x)
        x = self.addition(x, x)
        x = self.activation(x)
        return x

class Conv_layer(nn.Module):
    def __init__(self, item):
        super(Conv_layer, self).__init__()
        self.conv_layer = nn.Conv2d(in_channels=item['conv_num_ichan'], out_channels=item['conv_num_ochan'], kernel_size=item['conv_kernel_size'][0], stride=item['conv_strides'][0], padding=item['conv_pads'][0])
        self.bn_layer = nn.BatchNorm2d(num_features=item['conv_num_ochan'], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        
    def forward(self, x):
        x = self.conv_layer(x)
        x = self.bn_layer(x)
        return x
    
def MNSIMChiplet(layer, cycle, id):
    # get basic data 
    layer_type = layer['op_type']
    layer_model = layer['model']
    
    layer_params = []
    
    # generate layer parameters
    if layer_type == 'Conv-Act-Pool':
        layer_params.append(layer_model['conv_input_size'][0])
        layer_params.append(layer_model['conv_num_ichan'])
        layer_params.append(layer_model['conv_num_ochan'])
        layer_params.append(layer_model['conv_kernel_size'][0])
        layer_params.append(layer_model['pool_kernel_size'][0])
        layer_params.append(layer_model['pool_strides'][0])
        
    elif layer_type == 'Conv':
        layer_params.append(layer_model['conv_input_size'][0])
        layer_params.append(layer_model['conv_num_ichan'])
        layer_params.append(layer_model['conv_num_ochan'])
        layer_params.append(layer_model['conv_kernel_size'][0])
        if layer_model['conv_strides'][0] != 0:
        #     layer_params.append(layer_model['conv_pads'][0])
            layer_params.append(layer_model['conv_strides'][0])
        else:
            layer_params.append(None)
        #     layer_params.append(None)
            
    elif layer_type == 'Conv-Add-Act':
        layer_params.append(layer_model['conv_input_size'][0])
        layer_params.append(layer_model['conv_num_ichan'])
        layer_params.append(layer_model['conv_num_ochan'])
        layer_params.append(layer_model['conv_kernel_size'][0])
        layer_params.append(layer_model['conv_strides'][0])
        layer_params.append([-1, -2])
        
    elif layer_type == 'Conv-Act':
        layer_params.append(layer_model['conv_input_size'][0])
        layer_params.append(layer_model['conv_num_ichan'])
        layer_params.append(layer_model['conv_num_ochan'])
        layer_params.append(layer_model['conv_kernel_size'][0])
        
    print(layer_params)
    
    # generate weights
    if layer_type == 'Conv-Act-Pool':
        conv_structure = Conv_Act_Pool_layer(layer_model)
    
    elif layer_type == 'Conv-Add-Act':
        conv_structure = Conv_Add_Act_layer(layer_model)
        
    elif layer_type == 'Conv-Act':
        conv_structure = Conv_Act_layer(layer_model)
        
    elif layer_type == 'Conv':
        conv_structure = Conv_layer(layer_model)
        
    else:
        print("something wrong about model partition!")
        from IPython import embed; embed()
        
    layer_weight = conv_structure.state_dict()
    
    # set params, layer_type
    with open('./MNSIM/ParserConfig_' + str(id[0]) + '_' + str(id[1]) + '.ini', 'w') as file:
        # file.write(layer_type + '\n')
        for item in layer_params:
            file.write(str(item) + '\n')
            
    weight_path = './weight.pth'

    if os.path.exists(weight_path):
        os.remove(weight_path)
        
    torch.save(layer_weight, weight_path)
    
    for item in layer_weight:
        # print(layer_weight[item].size())
        print(item)
    
    # get data from sniper
    temporalModel(layer_type, weight_path, id)
    
    result_path = 'result_' + str(id[0]) + '_' + str(id[1]) + '.res'
    with open(result_path, 'r') as f:
        line = f.readline()
    
    # set cycle data
    print(float(line))
    time_cost = math.ceil(float(line) / cycle)
    # every cycle time_cost subtracting 1
    # transfer to a class? While time going becomes a function
    
    # run function model
    # model_input = []
    # ''' not finish here '''
    # model_output = conv_structure.forward(model_input)
        
    # return model_output
    return 1
    
# function to run function model
def functionModel(input_matrix, layer_type, params, weights_path):
    if (layer_type == 'conv'):
        # layer tpye: conv
        if (params[3] == None):
            layer = nn.Conv2d(in_channels=params[0], out_channels=params[1], kernel_size=params[2])
        elif (params[4] == None):
            layer = nn.Conv2d(in_channels=params[0], out_channels=params[1], kernel_size=params[2], padding=params[3])
        else:
            layer = nn.Conv2d(in_channels=params[0], out_channels=params[1], kernel_size=params[2], padding=params[3], stride=params[4])

        loaded_weights = torch.load(weights_path)
        
        layer.load_state_dict(loaded_weights)

        output_matrix = layer(input_matrix)
    elif (layer_type == 'relu'):
        # layer type: relu
        layer = nn.ReLU()
        
        output_matrix = layer(input_matrix)
    elif (layer_type == 'pooling'):
        # layer type: pooling
        layer = nn.MaxPool2d(kernel_size=params[0], stride=params[1])
        
        weights_path = 'your_weights.pth'
        loaded_weights = torch.load(weights_path)
        
        layer.load_state_dict(loaded_weights)
        
        output_matrix = layer(input_matrix)
    elif (layer_type == 'view'):
        # layer type: view
        output_matrix = input_matrix.view(1, -1)
    elif (layer_type == 'fc'):
        # layer type: fc
        layer = nn.Linear(in_features=params[1], out_features=params[2])
        
        weights_path = 'your_weights.pth'
        loaded_weights = torch.load(weights_path)
        
        layer.load_state_dict(loaded_weights)
        
        output_matrix = layer(input_matrix)
    elif (layer_type == 'dropout'):
        # alyer type: dropout
        layer = nn.Dropout(p=0.5) # set probability as 0.5
        
        output_matrix = layer(input_matrix)
    elif (layer_type == 'element_sum'):
        # layer type: element_sum
        output_matrix = torch.add(input_matrix[0], input_matrix[1]) # set number of papameters as 2
    else:
        1# error
    return output_matrix

# function to run temporal model
def temporalModel(layer_type, weights_path, id):
    print("[MNSIM  ]Start to run MNSIM2.0 chiplet.")
    
    mnsim_command = 'cd MNSIM; python ./main.py -NN ' + layer_type + ' -Weights .' + weights_path + ' -Id1 ' + str(id[0]) + ' -Id2 ' + str(id[1])
    os.system(mnsim_command)
    
    print("[MNSIM  ]Finish chiplet simulation.")

def readFromC(id):
    
    # with open('./MNSIMChiplet_' + str(id[0]) + '_' + str(id[1]) + '.cfg', "rb") as f:
    with open('/home/qc/Chiplet_Heterogeneous_newVersion_gem5/Chiplet_Heterogeneous_newVersion/tasks-vgg13/MNSIMChiplet_' + str(id[0]) + '_' + str(id[1]) + '.cfg', "rb") as f:
        cfg_dict = pickle.load(f)
        
    return cfg_dict
    
    # item['op_type'] = 'Conv-Add-Act'
    
    # model = {}
    
    # model['conv_num_ichan'] = 128
    # model['conv_num_ochan'] = 256
    # model['conv_kernel_size'] = [3, 3]
    # model['conv_strides'] = [2, 1]
    # model['conv_pads'] = [1, 1]
    
    # model['pool_kernel_size'] = [1, 2]
    # model['pool_strides'] = [1, 2]
    
    # item['model'] = model
    
def main():
    # get id
    parser = argparse.ArgumentParser(description='MNSIM Chiplet')
    parser.add_argument("-ID1", "--id1", default=0, help="index 1 of the chiplet")
    parser.add_argument("-ID2", "--id2", default=0, help="index 1 of the chiplet")
    
    args = parser.parse_args()
    
    id = [int(args.id1), int(args.id2)]
    
    # run simulation
    cfg_dict = readFromC(id)    
    
    MNSIMChiplet(cfg_dict, 1, id)
        
if __name__ == '__main__':
    main()
    # layer_type = 'Conv-Add-Act'
    # layer_params = [128, 128, 3, None, None]
    # layer_params = [128, 128, 3, 1, 1]
    # layer_index = [1, 2]
    
    # item = {}
    
    # # item['op_type'] = 'Conv-Act-Pool'
    # # item['op_type'] = 'Conv-Act'
    # item['op_type'] = 'Conv-Add-Act'
    # # item['op_type'] = 'Conv'
    
    # model = {}
    
    # model['conv_num_ichan'] = 64
    # model['conv_num_ochan'] = 64
    # model['conv_kernel_size'] = [3, 3]
    # model['conv_strides'] = [1, 1]
    # model['conv_pads'] = [1, 1]
    
    # model['pool_kernel_size'] = [2, 2]
    # model['pool_strides'] = [2, 2]
    # model['conv_input_size'] = [224, 224]
    
    # item['model'] = model
    
    
    # id = [0, 1]
    
    # cfg_dict = readFromC(id)    
    # t = MNSIMChiplet(cfg_dict, 1, id)
    # print(t)
    # main()