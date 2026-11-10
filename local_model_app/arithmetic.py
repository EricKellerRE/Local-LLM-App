"""Small deterministic arithmetic tool; no Python eval or arbitrary code."""
import ast
import math
import operator

SPEC = {"type":"function","function":{"name":"host__calculate",
    "description":"Execute arithmetic for unit conversions and derived metrics. Use this for ALL arithmetic; copy returned numeric results exactly.",
    "parameters":{"type":"object","properties":{"expressions":{"type":"object","minProperties":1,"maxProperties":16,
        "additionalProperties":{"type":"string","maxLength":256},
        "description":"Map descriptive result names to numeric expressions, e.g. scaled_value: 250 / 1000, product: 12 * 4."}},
        "required":["expressions"],"additionalProperties":False}}}
OPS={ast.Add:operator.add,ast.Sub:operator.sub,ast.Mult:operator.mul,ast.Div:operator.truediv}
def calculate(expressions):
    if not isinstance(expressions,dict) or not 1 <= len(expressions) <= 16:
        raise ValueError("Supply 1 to 16 named expressions.")
    def number(node):
        if isinstance(node,ast.Constant) and type(node.value) in (int,float):
            result=node.value
        elif isinstance(node,ast.BinOp) and type(node.op) in OPS:
            result=OPS[type(node.op)](number(node.left),number(node.right))
        elif isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.USub,ast.UAdd)):
            result=number(node.operand)*(-1 if isinstance(node.op,ast.USub) else 1)
        else:
            raise ValueError("Only numbers, parentheses, +, -, *, and / are supported.")
        if not math.isfinite(result) or abs(result)>1e100:
            raise ValueError("Result must be finite and within 1e100.")
        return result
    results={}
    for name,expression in expressions.items():
        if not isinstance(expression,str) or len(expression)>256:
            raise ValueError("Expressions must be strings up to 256 characters.")
        tree=ast.parse(expression,mode='eval')
        if len(list(ast.walk(tree)))>128:
            raise ValueError("Expression is too complex.")
        results[name]=number(tree.body)
    return results
