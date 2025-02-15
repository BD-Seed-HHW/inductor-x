
from utils import DEBUG_PRINT, IOType
import ast
import re

from utils import IOType

def get_kernel_param_list(kernel_param_list, signature_dict):
    kernel_inout_list = []
    for index in range(len(kernel_param_list)):
        param_str = kernel_param_list[index]
        if (index > len(signature_dict) or  ('*' not in signature_dict[index])):
            continue
        if ('_ptr' not in param_str):
            continue
        type = IOType.getIOType(param_str)
        kernel_inout_list.append(type)

    return kernel_inout_list

def extract_parameters(input_line):
    # 提取参数部分
    # 解析 triton_unk_fused_0.run(buf6, buf5, buf3, arg0_1, 3, 3, grid=grid(3, 3), stream=stream0)
    start = input_line.find('(')
    end = input_line.find(')')
    if start != -1 and end != -1:
        vars = (input_line[start + 1:end]).split(',')
        vars = [tmp.strip() for tmp in vars]
        kernel_name = input_line[0:start].split('.')[0].strip()
    else:
        raise ValueError(f"error parse_triton_function failed  line is:{input_line}")
    return kernel_name, vars

def parse_triton_function(line):
    # 解析 def triton_unk_fused_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, y0_numel, x1_numel,
    start = line.find('(')
    end = line.find(')')
    if start != -1 and end != -1:
        vars = (line[start + 1:end]).split(',')
        func_name = line[0:start].split(' ')[1].strip()
    else:
        raise ValueError(f"error parse_triton_function failed  line is:{line}")
    param_list = []
    for item in vars:
        tmp = item.strip()
        if ('ptr' in tmp):
            param_list.append(tmp)
    return func_name, param_list

def parse_signature(line):
    # 解析 triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'npu',
    # 'constants': {}, 'mix_mode': 'aiv',     'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},

    line = re.sub(r'^triton_meta=\s*|\s*,\s*$', '', line)
    signature_pattern = re.compile(r"'signature':\s*({[^}]*})")
    match = signature_pattern.search(line)
    if (match):
        signature_str = match.group(1)
        signature_dict = ast.literal_eval(signature_str)
        return signature_dict
    else:
        DEBUG_PRINT("error parse_signature failed line is:")
        DEBUG_PRINT(line)

def extract_triton_kernel_info(output_code_path):
    with open(output_code_path, 'r', encoding='utf-8') as source_file:
        lines = source_file.readlines()

    start_flag = False
    kernel_info = {}
    # 遍历triton kernel meta
    kernel_buf_len = {}
    for line in lines:
        # 1、 以async_compile.triton 为开始，2、 def triton_ 为结束，检索一个算子的输入输出、meta信息
        if ('async_compile.triton' in line):
            start_flag = True
        if ('triton_meta=' in line  and start_flag):
            signature_dict = parse_signature(line)
            start_flag = True
        elif ("def triton_" in line and start_flag):
            function_name, params = parse_triton_function(line)
            #DEBUG_PRINT(f"function_name={function_name}, params={params}")
            kernel_param_list = get_kernel_param_list(params, signature_dict)
            kernel_info[function_name] = kernel_param_list
            kernel_buf_len[function_name] = len(kernel_param_list)
            # 分割字符串
            signature_dict = None
            result_dict = None
            start_flag = False
    # 遍历 def call(args):
    kernel_detail = {}
    start_flag = False
    for line in lines:
        # 1、 以async_compile.triton 为开始，2、 def triton_ 为结束，检索一个算子的输入输出、meta信息
        if ('def call(args):' in line):
            start_flag = True
        elif ('.run(' in line and start_flag):
            kernel_name, params = extract_parameters(line)
            print(f"kernel_name={kernel_name}")
            if (kernel_name not in kernel_info):
                raise NotImplementedError("search kernel meta error")
            inout_map = {}
            kernel_name_param = kernel_info[kernel_name]

            for index in range(len(kernel_name_param)):
                key = params[index]
                inout_map[key] = kernel_name_param[index]

            if (kernel_name not in kernel_detail):
                kernel_detail[kernel_name] = inout_map
            else:
                old_map = kernel_detail[kernel_name]
                new_map = inout_map
                # 2次调用同一个kernel 变量一样，但是input output不一样
                for key, value in old_map.items():
                    if key in new_map and new_map[key] != value:
                        raise ValueError("kernel name inout_map parse failed")
                kernel_detail[kernel_name].update(inout_map)
        elif ('return (' in line):
            break

    return kernel_detail, kernel_buf_len

def extract_and_add_loadargs(dst_file, input_str, dump_path):
    parts = input_str.split('=')
    vars_part = parts[0].strip()
    #var_names = [var.strip() for var in vars_part.split(',')]
    var_names =  [part.strip() for part in vars_part.split(',') if part.strip()]

    result_dict = {}
    for var in var_names:
        result_dict[var] = dump_path + '/' + var

    char_lists = {var: list(var) for var in var_names}

    load_code = None
    for var in char_lists:
        tmp_line = f"    {var} = torch.load('{result_dict[var]}')\n"
        dst_file.write(tmp_line)

def insert_md5_line(destination_file, line, kernel_info, kernel_buf_len):
    # 解析字符串 分割字符变量 在run之前把input写进去  在run之后把output写进去
    kernel_name, params = extract_parameters(line)
    if ((kernel_name not in kernel_info) or (kernel_name not in kernel_buf_len)):
        raise NotImplementedError("search kernel meta error")
    inout_map = kernel_info[kernel_name]
    # 只需要循环最小的一个， len(inout_map)  和  len(params)
    input_md5_line = []
    output_md5_line = []
    lenth = kernel_buf_len[kernel_name]

    for index in range(lenth):
        # 生成input md5 line
        var = params[index] # 原始变量
        # 既是input 也是output 只需要保证output对齐即可
        if(IOType.isInput(inout_map[var]) and not IOType.isOutput(inout_map[var])):
            line1 = f"    tmp = hashlib.md5({var}.to(torch.float64).cpu().numpy().tobytes()).hexdigest()\n"
            input_md5_line.append(line1)
            line2 = f"    print(f'triton:{var}:{{tmp}}')\n"
            input_md5_line.append(line2)
        if (IOType.isOutput(inout_map[var])):
            line1 = f"    tmp = hashlib.md5({var}.to(torch.float64).cpu().numpy().tobytes()).hexdigest()\n"
            output_md5_line.append(line1)
            line2 = f"    print(f'triton:{var}:{{tmp}}')\n"
            output_md5_line.append(line2)

    for tmp_line in input_md5_line:
        destination_file.write(tmp_line)
    destination_file.write(line)
    for tmp_line in output_md5_line:
        destination_file.write(tmp_line)


from utils import write_lib,write_fixed_seed

def gen_output_code_hash(lines, fx_graph_dump_file, dump_path, kernel_info, kernel_buf_len):
    start_flag = False
    with open(fx_graph_dump_file, 'w', encoding='utf-8') as destination_file:
        destination_file.write("import hashlib\n")
        write_lib(destination_file)
        for line in lines:
            if ('def call(args):' in line):
                start_flag = True
                destination_file.write(line)
                write_fixed_seed(destination_file, 0)

            elif ('= args' in line and start_flag):
                destination_file.write(line)
                # load input的变量
                extract_and_add_loadargs(destination_file, line, dump_path)
            elif ('.run(' in line and start_flag):
                insert_md5_line(destination_file, line, kernel_info, kernel_buf_len)
            elif ('return print_performance' in line):
                destination_file.write("    times=1\n")
                destination_file.write("    repeat=1\n")
                destination_file.write(line)
            elif ('return ' in line):
                start_flag = False
                destination_file.write(line)
            else:
                # DEBUG_PRINT("error gen_fx_graph_template, line not contain = and torch.ops.aten.var_mean ")
                destination_file.write(line)


def insert_save_line(destination_file, line, kernel_info, kernel_buf_len, dump_path, dump_args):
    # 解析字符串 分割字符变量 在run之前把input写进去  在run之后把output写进去
    kernel_name, params = extract_parameters(line)
    if ((kernel_name not in kernel_info) or (kernel_name not in kernel_buf_len)):
        raise NotImplementedError("search kernel meta error")
    inout_map = kernel_info[kernel_name]
    # 只需要循环最小的一个， len(inout_map)  和  len(params)
    input_md5_line = []
    output_md5_line = []
    lenth = kernel_buf_len[kernel_name]


    for index in range(lenth):
        # 生成input md5 line
        var = params[index]  # 原始变量

        if (IOType.isInput(inout_map[var]) and var in dump_args):
            line1 = f"    torch.save({var}, '{dump_path}/output_code/{var}')\n"
            input_md5_line.append(line1)

        # 既是input 也是output 只需要保证output对齐即可
        if (IOType.isOutput(inout_map[var]) and var in dump_args):
            line1 = f"    torch.save({var}, '{dump_path}/output_code/{var}')\n"
            output_md5_line.append(line1)

    for tmp_line in input_md5_line:
        destination_file.write(tmp_line)

    destination_file.write(line)
    for tmp_line in output_md5_line:
        destination_file.write(tmp_line)


def gen_output_code_tensor(lines, fx_graph_dump_file, dump_path, kernel_info, kernel_buf_len, dump_args):
    start_flag = False
    with open(fx_graph_dump_file, 'w', encoding='utf-8') as destination_file:
        destination_file.write("import hashlib\n")
        destination_file.write("import torch\n")
        destination_file.write("import numpy\n")
        destination_file.write("import inductor_npu\n")
        destination_file.write("from torch import device\n")
        destination_file.write("import numpy as np\n")
        destination_file.write("import torch_npu\n")

        for line in lines:
            if ('def call(args):' in line):
                start_flag = True
                destination_file.write(line)
                write_fixed_seed(destination_file, 0)
            elif ('= args' in line and start_flag):
                destination_file.write(line)
                # load input的变量
                extract_and_add_loadargs(destination_file, line, dump_path)
            elif ('.run(' in line and start_flag):
                insert_save_line(destination_file, line, kernel_info, kernel_buf_len, dump_path, dump_args)
            elif ('return print_performance' in line):
                destination_file.write("    times=1\n")
                destination_file.write("    repeat=1\n")
                destination_file.write(line)
            elif ('return ' in line):
                start_flag = False
                destination_file.write(line)
            else:
                # DEBUG_PRINT("error gen_fx_graph_template, line not contain = and torch.ops.aten.var_mean ")
                destination_file.write(line)



def gen_output_code_template(output_code_path, dump_path, dump_type, dump_args):

    kernel_info, kernel_buf_len = extract_triton_kernel_info(output_code_path)
    with open(output_code_path, 'r', encoding='utf-8') as source_file:
        lines = source_file.readlines()

    if (dump_type == 'dump_hash'):
        fx_graph_dump_file = dump_path + '/output_code_dump_hash.py'
        gen_output_code_hash(lines, fx_graph_dump_file, dump_path, kernel_info, kernel_buf_len)

    elif (dump_type == 'dump_tensor'):
        fx_graph_dump_file = dump_path + '/output_code_dump_tensor.py'
        gen_output_code_tensor(lines, fx_graph_dump_file, dump_path, kernel_info, kernel_buf_len, dump_args)

    else:
        raise ValueError("error, dump_type not dump_hash or dump_hash")

    print(f"fx_graph_dump_path={fx_graph_dump_file}")
    return fx_graph_dump_file