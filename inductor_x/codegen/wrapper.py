from torch._inductor.codegen.wrapper import WrapperCodeGen, SymbolicCallArg
from torch._inductor.virtualized import V
class NPUWrapperCodeGen(WrapperCodeGen):
    def __init__(self):
        super().__init__()

    def write_triton_header_once(self) -> None:
        self.header.splice(
            """
            import triton
            import triton.language as tl
            from torch._inductor.triton_heuristics import split_scan_grid, start_graph, end_graph
            from inductor_npu.npu_triton_heuristics import grid
            {}
            """.format(
                V.graph.device_ops.import_get_raw_stream_as("get_raw_stream")
            )
        )

    #generate numel expr for range_tree_node
    def generate_node_numel_expr(self, kernel_name: str, node, numel_expr):
        expr = f"{kernel_name}_{node.name}_numel"
        if (expr, V.graph) not in self.kernel_numel_expr:
            # declare expr once in each graph (scope)
            self.kernel_numel_expr.add((expr, V.graph))
            self.writeline(
                f"{self.declare}{expr} = {self.expr_printer(numel_expr)}{self.ending}"
            )
        else:
            self.writeline(f"{expr} = {self.expr_printer(numel_expr)}{self.ending}")
        # We can get symbolic expressions here, like s0*64
        # It is fine to have them here, but we need to handle them correctly as their own type
        # This is tricky to do, so we wrap in a custom type, distinct from scalars, but also from sympy*
        # scalars as well.
        # This is handled in `generate_args_decl` which has a correct comment of: TODO: only works for
        # constant now, need type info. I agree, this needs type info, and while this is not true type info
        # it suffices as a type hint for the purposes of producing the correct code for this type.
        return SymbolicCallArg(expr, numel_expr)

    # don't free anything
    def make_buffer_free(self, buffer):
        #return f"del {buffer.get_name()}"
        return ""

    # don't assert
    def codegen_input_size_asserts(self) -> None:
        pass
