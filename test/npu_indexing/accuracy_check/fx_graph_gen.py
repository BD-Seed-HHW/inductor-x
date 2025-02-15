import re
from utils import DEBUG_PRINT


def is_tuple_result_op(line):
    tuple_result_op = {'torch.ops.aten.var_mean',
                       'torch.ops.aten.split',
                       'torch.ops.aten.convolution_backward',
                       'torch.ops.npu_graph.npu_fa',
                       'torch.ops.aten.nll_loss_forward',
                       'torch.ops.npu._npu_dropout'}
    for tmp in tuple_result_op:
        if (tmp in line):
            return True
    return False


def gen_fxgraph_dumporigin(lines, fx_graph_dump_path, dump_path, args):
    with open(fx_graph_dump_path, 'w', encoding='utf-8') as destination_file:
        destination_file.write("import hashlib\n")
        destination_file.write("import torch\n")
        destination_file.write("import numpy\n")
        destination_file.write("import torch_npu\n")
        destination_file.write("from torch import device\n")

        for line in lines:
            destination_file.write(line)

from utils import write_lib,write_fixed_seed

def gen_fxgraph_dumphash(lines, fx_graph_dump_path, dump_path, args):
    with open(fx_graph_dump_path, 'w', encoding='utf-8') as destination_file:
        destination_file.write("import hashlib\n")
        destination_file.write("import torch\n")
        destination_file.write("import numpy\n")
        destination_file.write("import torch_npu\n")
        destination_file.write("from torch import device\n")
        write_lib(destination_file)
        for line in lines:
            # if ('torch.ops.aten.split.Tensor' in line):
            #     import pdb;pdb.set_trace()
            if ('# File:' in line):
                destination_file.write(line)
                continue
            elif ('torch.nn.Module' in line):
                #import pdb; pdb.set_trace()
                if('<lambda>' in line):
                    # 替换 <lamda> 为GraphModule 并且保留前后空格、换行字符
                    new_line = line.replace('<lambda>', 'GraphModule')
                    destination_file.write(new_line)
                else:
                    destination_file.write(line)
            elif ('def forward(' in line):
                destination_file.write(line)

                for arg in args:
                    line1 = f"        tmp = hashlib.md5({arg['name']}.to(torch.float64).cpu().numpy().tobytes()).hexdigest()\n"
                    destination_file.write(line1)
                    new_line = f"        print(f'triton:{arg['name']}:{{tmp}}')\n"
                    destination_file.write(new_line)
                write_fixed_seed(destination_file, 1)


            elif (is_tuple_result_op(line)):
                destination_file.write(line)
                continue
            elif ('=' in line):
                # 提取 alias 和 div_1 变量名
                destination_file.write(line)
                alias_name = line.split(':')[0].strip()
                # 构造新的行内容，包括原始行和追加的哈希计算语句
                line1 = f"        tmp = hashlib.md5({alias_name}.to(torch.float64).cpu().numpy().tobytes()).hexdigest()\n"
                destination_file.write(line1)
                # if ('torch.ops.aten.add.Tensor' in line):
                #     new_line = f"        import pdb;pdb.set_trace()\n"
                #     destination_file.write(new_line)
                new_line= f"        print(f'triton:{alias_name}:{{tmp}}')\n"
                destination_file.write(new_line)
            else:
                destination_file.write(line)


def gen_fxgraph_dumptensor(lines, fx_graph_dump_file, dump_path, args, mid_tensor_args):
    with open(fx_graph_dump_file, 'w', encoding='utf-8') as destination_file:
        destination_file.write("import hashlib\n")
        destination_file.write("import torch\n")
        destination_file.write("import numpy\n")
        destination_file.write("import torch_npu\n")
        destination_file.write("from torch import device\n")
        write_lib(destination_file)
        for line in lines:
            # if ('torch.ops.aten.split.Tensor' in line):
            #     import pdb;pdb.set_trace()
            if ('# File:' in line):
                destination_file.write(line)
                continue
            elif ('torch.nn.Module' in line):
                #import pdb; pdb.set_trace()
                if('<lambda>' in line):
                    # 替换 <lamda> 为GraphModule 并且保留前后空格、换行字符
                    new_line = line.replace('<lambda>', 'GraphModule')
                    destination_file.write(new_line)
                else:
                    destination_file.write(line)
            elif ('def forward(' in line):
                destination_file.write(line)
                for var in args:
                    line1 = f"        torch.save({var}, '{dump_path}/fxgraph/{var}')\n"
                    destination_file.write(line1)
                write_fixed_seed(destination_file, 1)
            elif (is_tuple_result_op(line)):
                destination_file.write(line)
                continue
            elif ('=' in line):
                # 提取 alias 和 div_1 变量名
                destination_file.write(line)
                var = line.split(':')[0].strip()
                if (var in mid_tensor_args):
                    line1 = f"        torch.save({var}, '{dump_path}/fxgraph/{var}')\n"
                    destination_file.write(line1)

            else:
                destination_file.write(line)
def gen_fxgraph_template(fx_graph_readable_path, dump_path, dump_type,  input_args, mid_tensor_args):
    with open(fx_graph_readable_path, 'r', encoding='utf-8') as source_file:
        lines = source_file.readlines()

    fx_graph_dump_path = dump_path + '/fx_graph_dump_origin.py'
    gen_fxgraph_dumporigin(lines, fx_graph_dump_path, dump_path, input_args)

    if (dump_type == 'dump_hash'):
        fx_graph_dump_path = dump_path + '/fx_graph_dump_hash.py'
        gen_fxgraph_dumphash(lines, fx_graph_dump_path, dump_path,input_args)

    elif (dump_type == 'dump_tensor'):
        fx_graph_dump_path = dump_path + '/fx_graph_dump_tensor.py'
        gen_fxgraph_dumptensor(lines, fx_graph_dump_path, dump_path, input_args, mid_tensor_args)

    else:
        DEBUG_PRINT("error, dump_type not dump_hash or dump_hash")
        return
    DEBUG_PRINT(f"fx_graph_dump_path={fx_graph_dump_path}")
    return fx_graph_dump_path