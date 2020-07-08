import logging
import re
import time
import typing
from pathlib import Path

GLSL_IMPORT_DIR = "res/shaders/lib/"
GLSL_LIB_PATH = Path.cwd() / GLSL_IMPORT_DIR

# constants for group names
FUNC_RETURN_TYPE = "return_type"
FUNC_FUNCTION_NAME = "function_name"
FUNC_ARGUMENTS = "arguments"
IN_UNIFORM_ARGUMENT = "in_uniform_argument"

# REGEXP Definitions
REG_SPACE = re.compile(r"\s+", flags=re.UNICODE)
REG_GLSL_IMPORT = re.compile(r"\s*#\s*import\s*[\"'](\w*\.?\w*)[\"']", flags=re.UNICODE)
REG_GLSL_IN_OR_UNIFORM = re.compile(r"^\s*(?:in|uniform)\s+(?P<"
                                    + IN_UNIFORM_ARGUMENT + r">[\w_]+\s+[\w_]+)",
                                    flags=re.UNICODE | re.DOTALL | re.IGNORECASE)
REG_GLSL_FUNCTION_NAME = re.compile(r"^\s*(?:\w+)\s+([\w_]+)\(.*\)", flags=re.UNICODE | re.DOTALL | re.IGNORECASE)
REG_FUNCTION = re.compile(r"^\s*(?P<"
                          + FUNC_RETURN_TYPE + r">\w+)\s+(?P<"
                          + FUNC_FUNCTION_NAME + r">[\w_]+)\((?P<"
                          + FUNC_ARGUMENTS + r">[\w\s_,]*)\)",
                          flags=re.UNICODE | re.DOTALL | re.MULTILINE | re.IGNORECASE)
REG_LINE_COMMENT = re.compile(r"\s*\/\/.*", flags=re.UNICODE | re.DOTALL | re.IGNORECASE)

_logger = logging.getLogger(__name__)


def is_empty(s: str):
    """Returns True if the string does not contain any symbols, text or numbers, otherwise False."""
    return len(s.strip()) <= 0


def preprocess_imports(code: typing.List[str]) -> str:
    for i, line in enumerate(code):
        match = REG_GLSL_IMPORT.match(line)
        if match:
            filename = match.group(1)
            code[i] = get_import_code(filename)

    return "".join(code)


def get_import_code(import_file: str) -> str:
    lib_path = Path.cwd() / GLSL_IMPORT_DIR
    with open(lib_path / import_file, 'r') as f:
        libfile = f.read()
        return libfile


def generate_comment_line(comment: str):
    return "//---- {} ----".format(comment)


def replace_filename(line: str, new_name: str) -> str:
    match = REG_GLSL_FUNCTION_NAME.match(line)
    pos = match.regs[1]
    return line[0:pos[0]] + new_name + line[pos[1]:]


class GLSLCode:

    def __init__(self, code: typing.List[str], filename: str, primary_function: str):
        self.code = code
        self._generated_code = []
        self._node_num = -1
        self._reset = False
        self.filename = filename
        self.shader_name = self.filename.split("/")[-1].split(".")[0]
        self.functions = []
        self.imports = set()
        self.connected_code = set()
        self.parse_time = 0
        self._primary_func_name = primary_function
        self.primary_function: GLSLFunction = None
        self._uniforms = []

        self._arguments_to_add_to_primary = []

        # Track line numbers of important parts of code
        self._functions_start_line = -1
        self._primary_func_start_line = -1

        self._parse()
        self._set_primary_function()
        self._add_args_to_primary()

    def __str__(self):
        return "GLSLCode of {}".format(self.filename)

    def _parse(self):
        start_time = time.time()

        code_iterator = iter(self.code)
        for line_i, line in enumerate(code_iterator):

            if "/*" in line:
                while "*/" not in line:
                    line = next(code_iterator)  # Consume lines
                    line_i += 1

            # --- Import Statement ---
            match = REG_GLSL_IMPORT.match(line)

            if match:
                needed_file = match.group(1)
                self.imports.add(needed_file)
                continue

            # --- in or uniform ---
            match = REG_GLSL_IN_OR_UNIFORM.match(line)

            if match:
                arg = match.groupdict()[IN_UNIFORM_ARGUMENT]
                self._arguments_to_add_to_primary.append(arg)
                continue

            # --- Function ---
            match = REG_FUNCTION.match(line)

            if match:
                if not self._functions_start_line > 0:
                    self._functions_start_line = line_i  # Track which line the function definitions start on

                if not self._primary_func_start_line > 0:
                    func_name = match.groupdict()[FUNC_FUNCTION_NAME]
                    if func_name == self._primary_func_name:
                        self._primary_func_start_line = line_i

                func_code = [line]
                opening_braces = 0
                closing_braces = 0

                # Opening brace of primary_function might not be on the same line as the primary_function title!
                while not ("{" in line):
                    line = next(code_iterator)
                    func_code.append(line)

                opening_braces += 1

                while opening_braces > closing_braces:
                    line = next(code_iterator)
                    line_i += 1
                    func_code.append(line)

                    if "}" in line:
                        closing_braces += 1
                    if "{" in line:
                        opening_braces += 1

                self.functions.append(GLSLFunction(func_code, self))

        self.parse_time = time.time() - start_time

    def _add_args_to_primary(self):
        for arg_code in self._arguments_to_add_to_primary:
            self.primary_function.arguments.append(GLSLArgument(arg_code, self.primary_function))

    def _set_primary_function(self):
        name = self._primary_func_name
        for f in self.functions:
            if f.function_name == name:
                self.primary_function = f
                return

        raise KeyError("No primary_function with title {} in {}".format(name, self.filename))

    def reset(self, node_num: int):
        """
        Resets all connections for this code.
        :param node_num: The number of the node containing the code object.
        """
        self._node_num = node_num
        self._generated_code = []
        self.connected_code = set()
        self._uniforms = []
        for func in self.functions:
            func.reset()

        self._reset = True

    def get_node_num(self) -> int:
        return self._node_num

    def get_primary_function(self) -> 'GLSLFunction':
        return self.primary_function

    def get_modified_arg_name(self, arg: str) -> str:
        arg = self.primary_function.get_argument(arg)
        if arg:
            return arg.get_modified_name()

        return None

    def connect(self, argument: str, other_code: 'GLSLCode', out_arg: str = None):
        """
        Connect the argument 'argument' of this other_code's primary_function to the output of the primary_function of the other_code in
        'other_code'. If the primary function to be connected is using inout variables, specify the 'out_arg' as well.
        :param argument: The title of the argument that is to be replaced by a call to another primary_function.
        :param other_code: The other_code that will be called and whose return value will replace the argument
        :param out_arg: The name of the 'out' variable in the primary function.
        """
        self.connected_code.add(other_code)
        for sub_code in other_code.connected_code:
            self.connected_code.add(sub_code)

        self.primary_function.connect(argument, other_code, out_arg)

    def get_uniforms(self) -> typing.List[typing.Tuple[str, str]]:
        return self._uniforms

    def generate_code(self) -> str:
        if not self._reset:
            _logger.error("generate_code() called before the code has been reset. Call reset() first.")
            return ""

        # Step 1, copy all lines from the code up until the first function
        self._generated_code = self.code[0:self._functions_start_line].copy()

        # Step 2, insert uniforms for all connected nodes for each unconnected argument
        for needed_code in self.connected_code:
            needed_code._reset = False
            prim_func = needed_code.get_primary_function()
            for arg in prim_func.arguments:
                if not arg.is_connected() and arg.name != "frag_pos" and not arg.is_output():
                    self._generated_code.append(arg.get_uniform_string() + "\n")
                    self._uniforms.append((arg.type, arg.modified_name))

        # Step 3, handle imports
        all_imports = set([im for n in self.connected_code for im in n.imports])  # Convert to set to remove double imports
        for import_file in all_imports:
            libcode = generate_comment_line(import_file) + "\n" + get_import_code(import_file)
            self._generated_code.append(libcode + "\n")

        # Step 4, import code from connected nodes
        var_declarations = []
        function_calls = []
        added_function_defs = []
        _add_calls(self, var_declarations, function_calls, added_function_defs, self._generated_code)

        # for needed_code in self.connected_code:
        #     for needed_func in needed_code.functions:
        #         comment = generate_comment_line("{}.{}".format(needed_code.shader_name, needed_func.function_name))
        #
        #         if needed_func.modified_function_name not in added_function_defs:  # We don't want to add duplicate function definitions
        #             func_code = "\n" + comment + "\n" + "".join(needed_func.generated_code)
        #             self._generated_code.append(func_code + "\n")
        #             added_function_defs.append(needed_func.modified_function_name)
        #
        #         # Step 4.1, generate calls to connected functions
        #         func_calls, var_declars = needed_func.get_call_strings()
        #         var_declarations.extend(var_declars)
        #         function_calls.extend(func_calls)

        # Step 5, generate calls to connected functions in own primary function
        self.primary_function.add_calls(function_calls, var_declarations)
        primary_code = "".join(self.primary_function.generated_code)
        self._generated_code.append(primary_code)
        self._reset = False
        return "".join(self._generated_code)


def _add_calls(code: 'GLSLCode', declarations: list, calls: list, added_function_defs:list, generated_code: list):

    for arg in code.primary_function.arguments:
        if arg.is_connected() and arg.name != "frag_pos" and not arg.is_output():
            connected_code = arg.get_connected_func().get_parent_code()
            for needed_func in connected_code.functions:
                comment = generate_comment_line("{}.{}".format(connected_code.shader_name, needed_func.function_name))

                if needed_func.modified_function_name not in added_function_defs:  # We don't want to add duplicate function definitions
                    func_code = "\n" + comment + "\n" + "".join(needed_func.generated_code)
                    generated_code.append(func_code + "\n")
                    added_function_defs.append(needed_func.modified_function_name)

            _add_calls(connected_code, declarations, calls, added_function_defs, generated_code)

            # Step 4.1, generate calls to connected functions
            prim_function = connected_code.primary_function
            func_calls, var_declars = prim_function.get_call_strings()
            declarations.extend(var_declars)
            calls.extend(func_calls)


class GLSLFunction:

    def __init__(self, code: typing.List[str], parent_code: GLSLCode):
        self.code = code
        self._renamed_code = code.copy()
        self.generated_code = []
        self.parent_code = parent_code
        self.return_type = ""
        self.function_name = ""
        self.modified_function_name = ""
        self.arguments = []
        self._is_inout = False

        # Track line numbers of important parts of code
        self._body_start_line = -1

        self._parse()

    def __str__(self):
        return "(GLSLFunction) {} {}()".format(self.return_type, self.function_name)

    def _parse(self):

        for i, line in enumerate(self.code):
            # Parse first line
            match = REG_FUNCTION.match(line)

            if match:  # Function signature
                match_dict = match.groupdict()
                self.return_type = match_dict[FUNC_RETURN_TYPE]
                self.function_name = match_dict[FUNC_FUNCTION_NAME]

                # Rename the function so that we do not get any clashes
                self._set_modified_filename()
                new_func_def = replace_filename(self._renamed_code[i], self.modified_function_name)
                self._renamed_code[i] = new_func_def

                self._parse_arguments(match_dict[FUNC_ARGUMENTS])

            if "{" in line:
                self._body_start_line = i + 1
                break

        self.generated_code = self._renamed_code.copy()

    def _set_modified_filename(self):
        """Appends the filename to the function title to keep it unique."""
        if self.function_name == "main":  # We do not modify the title if it is the main function, as it is a reserved word
            self.modified_function_name = self.function_name
            return

        filename = self.parent_code.shader_name
        self.modified_function_name = filename + "_" + self.function_name

    def _parse_arguments(self, arguments: str):
        if is_empty(arguments):
            return

        args = arguments.split(",")

        for arg in args:
            a = GLSLArgument(arg, self)
            self.arguments.append(a)
            self._is_inout = self._is_inout or a.is_output()

    def reset(self):
        """Reset this function and all connections to it."""
        self.generated_code = self._renamed_code.copy()

        for arg in self.arguments:
            arg.reset()

    def is_inout(self):
        return self._is_inout

    def get_argument(self, name: str) -> 'GLSLArgument':
        for arg in self.arguments:
            if arg.name == name:
                return arg

        return None

    def get_output_args(self) -> typing.List['GLSLArgument']:
        args = []
        for arg in self.arguments:
            if arg.is_output():
                args.append(arg)

        return args

    def get_parent_code(self) -> 'GLSLCode':
        return self.parent_code

    def connect(self, argument: str, other_code: GLSLCode, out_argument: str = None):
        """
        Connects the output of the primary function of 'other_code' to the input with title 'argument'. If the primary function in the code to be
        connected is using inout variables, specify the connected output via 'out_arg'.
        :param argument: The title of the input argument of this code.
        :param other_code: The code whose output is to be input to the argument.
        :param out_argument: The name of the 'out' variable which is the output of the other code's primary function.
        """
        func = other_code.get_primary_function()
        try:
            arg = self.get_argument(argument)
            out_arg = func.get_argument(out_argument)
            arg.connect(func, other_arg=out_arg)
            if out_argument:
                func.get_argument(out_argument).connect(self, other_arg=arg)
        except AttributeError:
            raise KeyError("No argument with title {} in primary function {}".format(argument, func))

    def _get_call_string(self, argument: str) -> typing.Tuple[str, str]:
        declarations = []
        if self.is_inout():
            vars = []
            for i, arg in enumerate(self.arguments):
                if arg.is_output():
                    if arg.is_connected():
                        # Generate variable declaration like 'type name;' and add to call
                        other_arg = arg.get_connected_arg()
                        mod_other_arg = other_arg.get_modified_name()
                        declarations.append("\t{} {};".format(other_arg.type, mod_other_arg))
                        vars.append(mod_other_arg)
                    else:
                        mod_arg = arg.get_modified_name() + "_" + "UNUSED"
                        declarations.append("\t{} {};".format(arg.type, mod_arg))
                        vars.append(mod_arg)
                else:
                    vars.append(arg.get_modified_name())

            call = "\t{}({});".format(self.modified_function_name, ", ".join(vars))
        else:
            func = "{}({})".format(self.modified_function_name, ", ".join([arg.get_modified_name() for arg in self.arguments]))
            call = "\t{type} {arg} = {call};".format(type=self.return_type, arg=argument, call=func)

        declarations = "\n".join(declarations)
        return call, declarations

    def get_call_strings(self) -> typing.Tuple[typing.List[str], typing.List[str]]:
        var_declarations = []
        call_strings = []
        for arg in self.arguments:
            if arg.is_connected() and not arg.is_output():
                connected_func = arg._connected_func
                arg_call, arg_declarations = connected_func._get_call_string(arg.get_modified_name())
                var_declarations.append(arg_declarations + "\n")
                call_strings.append(arg_call + "\n")

        return call_strings, var_declarations

    def add_calls(self, calls: typing.List[str], var_declars: typing.List[str]):
        own_call_strings, own_variable_declarations = self.get_call_strings()
        var_declars.insert(0, generate_comment_line("Variable Declarations") + "\n")
        var_declars.extend(own_variable_declarations)
        calls.insert(0, generate_comment_line("Function Calls") + "\n")
        calls.extend(own_call_strings)

        if len(calls) > 0:
            insert_line = self._body_start_line
            declarations_code = "".join(var_declars)
            calls_code = "".join(calls)
            self.generated_code.insert(insert_line, declarations_code + "\n" + calls_code)


class GLSLArgument:

    def __init__(self, code: str, parent_function: GLSLFunction):
        self.code = code.strip()
        self.parent_function = parent_function
        self.type = ""
        self.name = ""
        self.inout = ""
        self.modified_name = ""
        self._connected_func = None
        self._connected_arg = None
        self._is_connected = False

        self._parse()

    def __str__(self):
        if self.inout:
            return "(GLSLArgument) {} {} {}".format(self.inout, self.type, self.name)
        else:
            return "(GLSLArgument) {} {}".format(self.type, self.name)

    def _parse(self):
        parts = REG_SPACE.split(self.code)
        if len(parts) == 2:
            self.type = parts[0]
            self.name = parts[1]
        elif len(parts) == 3:
            self.inout = parts[0]
            self.type = parts[1]
            self.name = parts[2]
        else:
            raise SyntaxError("GLSLArguments needs to have at least a type and a name but was given: {}".format(parts))

    def get_node_num(self):
        return self.parent_function.parent_code.get_node_num()

    def get_modified_name(self) -> str:
        if self.name == "frag_pos" or self.parent_function.function_name == "main":
            return self.name
        else:
            return "{}_{}_{}".format(self.parent_function.modified_function_name, self.get_node_num(), self.name)

    def get_parent_code(self) -> 'GLSLCode':
        return self.parent_function.parent_code

    def get_connected_func(self) -> 'GLSLFunction':
        return self._connected_func

    def get_connected_arg(self) -> 'GLSLArgument':
        return self._connected_arg

    def is_connected(self) -> bool:
        return self._is_connected

    def is_output(self) -> bool:
        return "out" in self.inout

    def reset(self):
        self._is_connected = False
        self._connected_func = None

    def get_uniform_string(self) -> str:
        return "uniform {} {};".format(self.type, self.get_modified_name())

    def get_declaration_string(self) -> str:
        return "{} {};".format(self.type, self.get_modified_name())

    def connect(self, other_func: GLSLFunction, other_arg: 'GLSLArgument' = None):
        """Connects this argument to a call to another shaders primary_function or directly to another argument of another function,
        if inout arguments are used."""
        self._connected_func = other_func
        self._connected_arg = other_arg
        self._is_connected = True
