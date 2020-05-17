import re
import time
import typing
from pathlib import Path

GLSL_IMPORT_DIR = "res/lib/"
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
        self.filename = filename
        self.shader_name = self.filename.split(".")[0]
        self.functions = []
        self.imports = set()
        self.needed_code = set()
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

                # Opening brace of primary_function might not be on the same line as the primary_function name!
                while not ("{" in line):
                    line = next(code_iterator)
                    func_code.append(line)

                opening_braces += 1

                while opening_braces > closing_braces:
                    line = next(code_iterator)
                    func_code.append(line)

                    if "}" in line:
                        closing_braces += 1
                    if "{" in line:
                        opening_braces += 1

                self.functions.append(GLSLFunction(func_code, self))

        self.parse_time = time.time() - start_time

    def _add_args_to_primary(self):
        for arg_code in self._arguments_to_add_to_primary:
            self.primary_function.arguments.append(GLSLVariable(arg_code, self.primary_function))

    def _set_primary_function(self):
        name = self._primary_func_name
        for f in self.functions:
            if f.function_name == name:
                self.primary_function = f
                return

        raise KeyError("No primary_function with name {} in {}".format(name, self.filename))

    def reset(self):
        """Resets all connections for this code."""
        self._generated_code = []
        self.needed_code = set()
        self._uniforms = []
        for func in self.functions:
            func.reset()

    def get_primary_function(self) -> 'GLSLFunction':
        return self.primary_function

    def connect(self, argument: str, other_code: 'GLSLCode'):
        """
        Connect the argument 'argument' of this other_code's primary_function to the output of the primary_function of the other_code in 'other_code'.
        :param argument: The name of the argument that is to be replaced by a call to another primary_function.
        :param other_code: The other_code that will be called and whose return value will replace the argument
        :return:
        """
        self.needed_code.add(other_code)
        for sub_code in other_code.needed_code:
            self.needed_code.add(sub_code)

        self.primary_function.connect(argument, other_code)

    def get_uniforms(self) -> typing.List[typing.Tuple[str, str]]:
        return self._uniforms

    def generate_code(self) -> str:
        # Step 1, copy all lines from the code up until the first function
        self._generated_code = self.code[0:self._functions_start_line].copy()

        # Step 2, insert uniforms for all connected nodes for each unconnected argument
        for prim_func in [c.primary_function for c in self.needed_code]:
            for arg in prim_func.arguments:
                if not arg.is_connected() and arg.name != "vert_pos":
                    self._generated_code.append(arg.get_uniform_string() + "\n")
                    self._uniforms.append((arg.type, arg.modified_name))

        # Step 3, handle imports
        all_imports = [im for n in self.needed_code for im in n.imports]
        for import_file in self.imports:
            libcode = generate_comment_line(import_file) + "\n" + get_import_code(import_file)
            self._generated_code.append(libcode + "\n")

        # Step 4, import code from connected nodes
        function_calls = []
        for needed_code in self.needed_code:
            for needed_func in needed_code.functions:
                comment = generate_comment_line("{}.{}".format(needed_code.shader_name, needed_func.function_name))

                # Step 2.1, generate calls to connected functions
                function_calls.extend(needed_func.get_call_strings())

                func_code = "\n" + comment + "\n" + "".join(needed_func.generated_code)
                self._generated_code.append(func_code + "\n")

        # Step 5, generate calls to connected functions in own primary function
        self.primary_function.add_function_calls(function_calls)
        primary_code = "".join(self.primary_function.generated_code)
        self._generated_code.append(primary_code)

        return "".join(self._generated_code)


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
        """Appends the filename to the function name to keep it unique."""
        if self.function_name == "main":  # We do not modify the name if it is the main function, as it is a reserved word
            self.modified_function_name = self.function_name
            return

        filename = self.parent_code.shader_name
        self.modified_function_name = filename + "_" + self.function_name

    def _parse_arguments(self, arguments: str):
        if is_empty(arguments):
            return

        vars_ = arguments.split(",")

        for var in vars_:
            self.arguments.append(GLSLVariable(var, self))

    def reset(self):
        """Reset this function and all connections to it."""
        self.generated_code = self._renamed_code.copy()

        for arg in self.arguments:
            arg.reset()

    def connect(self, argument: str, other_code: GLSLCode):
        func = other_code.get_primary_function()
        for arg in self.arguments:
            if argument == arg.name:
                arg.connect(func)
                return

        raise KeyError("No argument with name {} in primary function {}".format(argument, func))

    def _get_call_string(self, argument: str) -> str:
        call = "{}({})".format(self.modified_function_name, ", ".join([arg.modified_name for arg in self.arguments]))
        return "\t{type} {var} = {call};".format(type=self.return_type, var=argument, call=call)

    def get_call_strings(self) -> typing.List[str]:
        call_strings = []
        for arg in self.arguments:
            if arg.is_connected():
                connected_func = arg.connected_to
                call_string = connected_func._get_call_string(arg.modified_name) + "\n"
                call_strings.append(call_string)

        return call_strings

    def add_function_calls(self, calls: typing.List[str]):
        own_call_strings = self.get_call_strings()
        calls.extend(own_call_strings)

        if len(calls) > 0:
            insert_line = self._body_start_line
            calls_code = "".join(calls)
            self.generated_code.insert(insert_line, calls_code)


class GLSLVariable:

    def __init__(self, code: str, parent_function: GLSLFunction):
        self.code = code.strip()
        self.parent_function = parent_function
        self.type = ""
        self.name = ""
        self.modified_name = ""
        self.connected_to = None
        self._is_connected = False

        self._parse()

    def __str__(self):
        return "(GLSLVariable) {} {}".format(self.type, self.name)

    def _parse(self):
        parts = REG_SPACE.split(self.code)
        self.type = parts[0]
        self.name = parts[1]
        self.modified_name = self._get_modified_name()

    def _get_modified_name(self) -> str:
        if self.name == "vert_pos" or self.parent_function.function_name == "main":
            return self.name
        else:
            return self.parent_function.modified_function_name + "_" + self.name

    def is_connected(self) -> bool:
        return self._is_connected

    def reset(self):
        self._is_connected = False
        self.connected_to = None

    def get_uniform_string(self) -> str:
        return "uniform {} {};".format(self.type, self.modified_name)

    def connect(self, other: GLSLFunction):
        """Connects this variable to a call to another shaders primary_function."""
        self.connected_to = other
        self._is_connected = True
