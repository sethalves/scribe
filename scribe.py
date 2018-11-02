import sys
import os
import time
import re
import argparse

def coerceToDict(value):
    if value == None: return None
    if isinstance(value, dict): return value
    if isinstance(value, str): return { value: True }
    if isinstance(value, list):
        result = {}
        for v in value: result[v] = True
        return result
    raise ValueError("Unsupported value")

class Function:
    NULL_VALUE = '_SCRIBE_NULL'
    @staticmethod
    def parseFunction(functionString):
        args = []
        m = re.search(r'(\w+)\((.*)\)', functionString)
        if not m:
            raise RuntimeError('Invalid function string "{}"'.format(functionString))
        if m.group(2) != '':
            args = re.compile(r',\s*').split(m.group(2))
        return [m.group(1), args]

    def __init__(self, functionString, functionBody):
        (self.name, self.args) = Function.parseFunction(functionString)
        self.block = Block(functionBody)

    def parseArguments(self, arguments, context):
        index = 0
        maxIndex = min(len(arguments), len(self.args))
        result = {}
        while index < maxIndex:
            value = arguments[index]
            m = re.match(r'\$(\w+)\$', value)
            if m: value = context.evaluateFunctionParam(m.group(1))
            if Function.NULL_VALUE == value: value = None
            if value != None: result[self.args[index]] = value
            index = index + 1
        return result

class Block:
    END_TAG_RE = re.compile(r'([!@$])>', flags=re.MULTILINE)
    START_TAG_RES = {
        '!':re.compile(r'<!', flags=re.MULTILINE),
        '@':re.compile(r'<@', flags=re.MULTILINE),
        '$':re.compile(r'<\$', flags=re.MULTILINE),
    }
    IF_ENDIF_TAGS = ['if', 'endif']
    FUNC_ENDFUNC_TAGS = ['func', 'endfunc']
    ENDIF_DEEP_CANDIDATES = ['if', 'endif']
    ENDIF_SHALLOW_CANDIDATES = ['if', 'endif', 'else', 'elif']

    def __init__(self, source):
        if isinstance(source, str):
            self.parsed = []
            # find an end tag
            me = re.search(Block.END_TAG_RE, source)
            while me is not None:
                startPattern = Block.START_TAG_RES[me.group(1)]
                mb = re.search(startPattern, source)
                if mb.start() > 0:
                    self.parsed.append(source[:mb.start()])
                self.parsed.append([me.group(1), source[mb.end():me.start()]])
                source = source[me.end():]
                me = re.search(Block.END_TAG_RE, source)
            self.parsed.append(source)
        elif isinstance(source, list):
            self.parsed = source.copy()
        else: 
            raise RuntimeError()
        if not self.isBalanced():
            raise ValueError("Block does not have balanced if/endif directives")
        self.stripComments()

    def copy(self):
        return Block(self.parsed)
        
    def cleanLines(self, lines):
        result = ''.join(lines)
        result = re.sub(r'\n\s*\n', '\n', result, flags = re.MULTILINE)
        return result

    def toString(self):
        return ''.join(self.parsed)

    def clean(self):
        newParsed = []
        lastIndex = 0
        (index, entry) = self.findNextDirective(lastIndex)
        while index < len(self.parsed):
            newParsed.append(self.cleanLines(self.parsed[lastIndex:index]))
            newParsed.append(self.parsed[index])
            lastIndex = index + 1
            (index, entry) = self.findNextDirective(lastIndex)
        newParsed.append(self.cleanLines(self.parsed[lastIndex:]))
        self.parsed = newParsed

    def findNextDirective(self, index, dtype = None):
        while index < len(self.parsed):
            entry = self.parsed[index]
            if not isinstance(entry, list):
                index += 1
                continue
            if dtype != None and entry[0] != dtype:
                index += 1
                continue
            return [index, entry]
        return [index, None]

    def findNextCommand(self, index, candidates = None):
        candidates = coerceToDict(candidates)
        (index, entry) = self.findNextDirective(index, '@')
        while index < len(self.parsed):
            commands = entry[1].split()
            if (candidates is None) or (commands[0] in candidates):
                return [index, commands]
            (index, entry) = self.findNextDirective(index + 1, '@')
        return [index, None]

    def isBalanced(self):
        ifs = 0
        endifs = 0
        index = 0
        (index, commands) = self.findNextCommand(0, Block.IF_ENDIF_TAGS)
        while index < len(self.parsed):
            if commands[0] == 'if':
                ifs = ifs + 1
            elif commands[0] == 'endif':
                endifs = endifs + 1
            else:
                raise RuntimeError()
            (index, commands) = self.findNextCommand(index + 1, Block.IF_ENDIF_TAGS)
        return ifs == endifs

    def replace(self, beginIndex, endIndex, replaceWith = []):
        newList = self.parsed[0:beginIndex]
        newList.extend(replaceWith)
        newList.extend(self.parsed[endIndex:])
        self.parsed = newList

    # Stand the parsed command list for the else/elif/endif lines of the current 
    # block, skipping any sub-blocks
    def findEndfunc(self, startIndex):
        depth = 0
        (index, commands) = self.findNextCommand(startIndex, Block.FUNC_ENDFUNC_TAGS)
        while index < len(self.parsed):
            if (depth == 0) and (commands[0] == 'endfunc'):
                return index
            if commands[0] == 'func':
                depth = depth + 1
            elif commands[0] == 'endfunc':
                depth = depth - 1
            (index, commands) = self.findNextCommand(index + 1, Block.FUNC_ENDFUNC_TAGS)
        raise RuntimeError("Unable to find endfunc matching {}".format(self.parsed[startIndex]))

    def stripComments(self):
        (index, entry) = self.findNextDirective(0, '!')
        while index < len(self.parsed):
            del self.parsed[index]
            (index, entry) = self.findNextDirective(index, '!')

    # Stand the parsed command list for the else/elif/endif lines of the current 
    # block, skipping any sub-blocks
    def findEndif(self, index):
        candidates = Block.ENDIF_SHALLOW_CANDIDATES
        depth = 0
        result = []
        (index, commands) = self.findNextCommand(index, candidates)
        while index < len(self.parsed):
            if (depth == 0) and (commands[0] == 'endif'):
                result.append([index, commands])
                return result
            if commands[0] == 'if':
                depth = depth + 1
                candidates = Block.ENDIF_DEEP_CANDIDATES
            elif commands[0] == 'endif':
                depth = depth - 1
                if 0 == depth:
                    candidates = Block.ENDIF_SHALLOW_CANDIDATES
            elif commands[0] == 'else' or commands[0] == 'elif':
                result.append([index, commands])
            (index, commands) = self.findNextCommand(index + 1, candidates)
        raise RuntimeError("Unable to find endif matching ...")

    def evaluateIfdef(self, conditionals, context):
        index = 0
        while index < (len(conditionals) - 1):
            conditional = conditionals[index]
            commands = conditional[1]
            if commands[0] == 'else':
                # no valid block found, but got to an else clause
                return [True, index]
            # evaluate the expression and possibly return 
            elif (commands[0] == 'if' or commands[0] == 'elif') and context.evaluateAsBoolean(commands[1:]):
                return [True, index]
            index = index + 1
        # none of the blocks evaluated to true
        return [False, index]

    def processDirectives(self, context):
        candidates = ['if', 'def', 'func', 'include']
        (index, commands) = self.findNextCommand(0, candidates)
        while index < len(self.parsed):
            if commands[0] == 'def':
                name = commands[1]
                value = True
                if len(commands) > 2: value = ' '.join(commands[2:])
                context.define(name, value)
                del self.parsed[index]
            elif commands[0] == 'if':
                conditionals = self.findEndif(index + 1)
                conditionals.insert(0, [index, commands])
                endifIndex = conditionals[len(conditionals) - 1][0]
                # Does one of the conditional blocks need to be kept?
                (found, validIndex) = self.evaluateIfdef(conditionals, context)
                keepBlock = []
                if found:
                    keepStartIndex = conditionals[validIndex][0]
                    keepEndIndex = conditionals[validIndex + 1][0]
                    keepBlock = self.parsed[keepStartIndex + 1:keepEndIndex].copy()
                self.replace(index, endifIndex + 1, keepBlock)
            elif commands[0] == 'func':
                endIndex = self.findEndfunc(index + 1)
                name = ' '.join(commands[1:])
                body = self.parsed[index + 1: endIndex]
                context.addFunction(Function(name, body))
                self.replace(index, endIndex + 1)
            elif commands[0] == 'include':
                childBlock = context.resolveInclude(commands[1])
                childBlock.resolve(context)
                self.replace(index, index + 1, childBlock.parsed)
            else:
                raise RuntimeError("Unreachable")
            (index, commands) = self.findNextCommand(index, candidates)

    def resolve(self, context):
        self.stripComments()
        self.processDirectives(context)
        self.clean()
        self.replaceVars(context)

    def replaceVars(self, context):
        (index, entry) = self.findNextDirective(0, '$')
        while index < len(self.parsed):
            result = context.evaluateVar(entry[1])
            self.replace(index, index + 1, result)
            (index, entry) = self.findNextDirective(index + 1, '$')

class Scribe:
    SHADER_TYPE_MAP = {
        'vert': 'GPU_VERTEX_SHADER',
        'frag': 'GPU_PIXEL_SHADER',
        'geom': 'GPU_GEOMETRY_SHADER',
        'comp': 'GPU_COMPUTE_SHADER',
    }

    #
    def __init__(self, shaderType, includeDirs=[], headers=[], input=sys.stdin, output=sys.stdout, makefile=False, defines=[]):
        self.input = input
        self.output = []
        self.headers = headers
        self.shaderType = shaderType
        self.includeDirs = includeDirs
        self.type = shaderType
        self.makefile = makefile
        self.stack = []
        self.defines = {}
        self.functions = {}
        self.includes = {}
        self.root = Block(self.input.read())
        for define in defines:
            if len(define) == 1:
                self.defines[define[0]] = True
            elif len(define) == 2:
                self.defines[define[0]] = define[1]
            else:
                raise RuntimeError()
        # FIXME 
        self.defines['_SCRIBE_DATE'] = 'Now'

    def getFunction(self, name):
        if name not in self.functions:
            raise RuntimeError("bad function name {}".format(name))
        return None

    def addFunction(self, function):
        self.functions[function.name] = function

    def define(self, name, value):
        self.defines[name] = value
        
    def findIncludePath(self, path):
        for includeDir in self.includeDirs:
            fullpath = os.path.join(includeDir, path)
            if os.path.exists(fullpath):
                self.includes[path] = fullpath
                return fullpath
        raise RuntimeError("Could not find {} in include paths: {}".format(path, self.includes))

    def resolveInclude(self, path):
        while path in self.includes:
            resolved = self.includes[path]
            # if the resolved value is another string, restart the search
            if isinstance(resolved, str):
                path = resolved
                continue
            return resolved.copy()

        fullpath = self.findIncludePath(path)
        with open(fullpath) as f:
            self.includes[fullpath] = Block(f.read())
        return self.includes[fullpath].copy()
        
    def pushStack(self, args):
        self.stack.insert(0, args)

    def popStack(self):
        self.stack.pop(0)

    def evaluateFunctionParam(self, param):
        for frame in self.stack:
            if param in frame:
                return frame[param]
        return None
        #raise RuntimeError("Unable to resolve function parameter {}".format(param))

    def evaluateFunction(self, name, args):
        if name not in self.functions:
            raise RuntimeError('Uknown function "{}"'.format(name))
        function = self.functions[name]
        args = function.parseArguments(args, self)
        body = function.block.copy()
        self.pushStack(args)
        body.resolve(self)
        self.popStack()
        return body.parsed

    def evaluateVar(self, variable):
        if -1 != variable.find(')'):
            (name, args) = Function.parseFunction(variable)
            return self.evaluateFunction(name, args)
        for frame in self.stack:
            if variable in frame:
                return [frame[variable]]
        if variable in self.defines:
            return [self.defines[variable]]
        return []

    def evaluateAsBoolean(self, expression):
        if expression[0] == 'not':
            return not self.evaluateAsBoolean(expression[1:])
        if len(expression) == 3:
            (var, operator, val) = expression
            resolvedVar = None
            if var in self.defines:
                resolvedVar = self.defines[var]
            if operator == '==':
                return resolvedVar == val
            elif operator == '<':
                return resolvedVar < val
            elif operator == '>':
                return resolvedVar > val
            else:
                raise RuntimeError("Invalid operator {}".format(operator))
        for frame in self.stack:
            if expression[0] in frame:
                return True
        if (expression[0] == '1') or (expression[0] == 'true'):
            return True
        if expression[0] in self.defines:
            return True
        return False

    def process(self):
        self.root.resolve(self)
        # Makefile mode... only output the included files
        if self.makefile:
            result = []
            for key in self.includes.keys():
                if isinstance(self.includes[key], str):
                    continue
                result.append(key)
            self.output.append('\n'.join(result))
            self.output.append('\n')
        else:
            for header in self.headers:
                with open(header) as f:
                    self.output.append(f.read())
                    self.output.append('\n')
            self.output.append('#define {}\n'.format(Scribe.SHADER_TYPE_MAP[self.shaderType]))
            self.output.append(self.root.toString())
        return ''.join(self.output)

def parseArguments():
    parser = argparse.ArgumentParser(description='Scribe shaders')
    parser.add_argument('--output', '-o', nargs='?', type=argparse.FileType('w'), default=sys.stdout)
    parser.add_argument('--input', '-i', nargs='?', type=argparse.FileType('r'), default=sys.stdin)
    parser.add_argument('--define', '-D', type=str, nargs=2, action='append')
    parser.add_argument('--type', '-T', type=str, default='vert', action='store', choices=['vert','frag','geom', 'comp'],)
    parser.add_argument('--include', '-I', type=str, action='append')
    parser.add_argument('--header', '-H', type=str, action='append')
    parser.add_argument('--makefile', '-M', action='store_true')
    # Check for debug mode
    if 1 == len(sys.argv):
        sourceDir = os.path.expanduser('~/git/hifi')
        testIncludeLibs = ['gpu', 'render', 'graphics', 'display-plugins', 'procedural', 'render-utils']
        testArgs = [
            '-D', 'GLPROFILE', 'PC_GL', 
            '-T', 'frag', 
            # makefile mode
            #'-M',
            '-H', 'C:/Users/bdavi/git/hifi/libraries/shaders/headers/410/header.glsl', 
            '-H', 'C:/Users/bdavi/git/hifi/libraries/shaders/headers/mono.glsl', 
            '-i', 'C:\\Users\\bdavi/git/hifi/libraries/render-utils/src/model_lightmap.slf', 
            #'-o', 'C:/Users/bdavi/git/hifi/build_noui/libraries/shaders/shaders/gpu/410/DrawTexcoordRectTransformUnitQuad.vert'
            '-o', 'D:/Shaders/temp.frag'
        ]
        for lib in testIncludeLibs:
            testArgs.extend(['-I', os.path.join(sourceDir, 'libraries/{}/src/{}/'.format(lib, lib))])
            testArgs.extend(['-I', os.path.join(sourceDir, 'libraries/{}/src/'.format(lib))])
        return parser.parse_args(testArgs)
    return  parser.parse_args()

if __name__ == "__main__":
    args = parseArguments()
    scribe = Scribe(
        shaderType=args.type, 
        input=args.input, 
        includeDirs=args.include, headers=args.header, defines=args.define,
        makefile=args.makefile)
    result = scribe.process()
    args.output.write(result)
