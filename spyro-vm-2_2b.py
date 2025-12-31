#!/usr/bin/env python3
# Spyro VM 2.2-B
# - Based on 2.1-D (modules, exceptions, OO)
# - Adds a Spyro-managed heap (Arena + Obj + Ref) and mark-and-sweep GC.
#
# Design follows classic mark-sweep:
# - Roots: VM stack, module globals, call frames, open upvalues, etc.
# - Mark phase traces object graph; sweep frees unmarked objs; next_gc threshold grows. [page:1][page:0]

from __future__ import annotations

import os
import sys
import json as py_json
import time as py_time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Iterable

# =====================================
# Heap (Arena + Obj + Ref) + GC
# =====================================

@dataclass(frozen=True)
class Ref:
    oid: int

class ObjKind(str, Enum):
    STRING = "string"
    LIST = "list"
    DICT = "dict"
    CLOSURE = "closure"
    UPVALUE = "upvalue"
    CLASS = "class"
    INSTANCE = "instance"
    BOUND_METHOD = "bound_method"

@dataclass
class Obj:
    oid: int
    kind: ObjKind
    marked: bool = False
    payload: Any = None

    def children(self) -> Iterable[Any]:
        """
        Return values referenced by this object (which may include Ref or immediate values).
        Marking logic will mark only Refs (immediates are ignored). [page:1]
        """
        k = self.kind
        p = self.payload
        if k == ObjKind.STRING:
            return ()
        if k == ObjKind.LIST:
            return p["items"]
        if k == ObjKind.DICT:
            # mark keys and values (keys can be strings or others)
            out = []
            for kk, vv in p["items"].items():
                out.append(kk)
                out.append(vv)
            return out
        if k == ObjKind.CLOSURE:
            return [p["function_ref"]] + list(p["upvalues"])
        if k == ObjKind.UPVALUE:
            return [p["closed"]] if p["is_open"] is False else []
        if k == ObjKind.CLASS:
            out = []
            if p["superclass"] is not None:
                out.append(p["superclass"])
            for _, mref in p["methods"].items():
                out.append(mref)
            return out
        if k == ObjKind.INSTANCE:
            out = [p["klass"]]
            for _, vv in p["fields"].items():
                out.append(vv)
            return out
        if k == ObjKind.BOUND_METHOD:
            return [p["receiver"], p["method"]]
        return ()

class Arena:
    def __init__(self):
        self.objs: Dict[int, Obj] = {}
        self.next_id = 1

        # accounting / threshold strategy inspired by bytesAllocated/nextGC. [page:1]
        self.alloc_count = 0
        self.next_gc = 2000  # count-based threshold (simple)
        self.last_collected = 0

        self.trace_gc = False

    def alloc(self, kind: ObjKind, payload: Any) -> Ref:
        oid = self.next_id
        self.next_id += 1
        self.objs[oid] = Obj(oid=oid, kind=kind, payload=payload, marked=False)
        self.alloc_count += 1
        if self.trace_gc:
            print(f"[gc] alloc oid={oid} kind={kind}")
        return Ref(oid)

    def get(self, r: Ref) -> Obj:
        return self.objs[r.oid]

    def maybe_collect(self, vm: "VM"):
        if self.alloc_count >= self.next_gc:
            self.collect(vm)

    def collect(self, vm: "VM"):
        if self.trace_gc:
            print("[gc] -- gc begin --")
            print(f"[gc] objects={len(self.objs)} alloc_count={self.alloc_count} next_gc={self.next_gc}")

        # mark phase
        gray: List[int] = []
        def mark_value(v: Any):
            if isinstance(v, Ref):
                mark_obj(v.oid)

        def mark_obj(oid: int):
            obj = self.objs.get(oid)
            if obj is None:
                return
            if obj.marked:
                return
            obj.marked = True
            gray.append(oid)
            if self.trace_gc:
                print(f"[gc] mark oid={oid} kind={obj.kind}")

        # roots (stack/globals/frames/open upvalues/modules). [page:1]
        for v in vm.stack:
            mark_value(v)

        # module globals (all dict values; keys are python strings)
        for g in vm.modules.values():
            for _, v in g.items():
                mark_value(v)

        # root globals too (entry module)
        for _, v in vm.root_globals.items():
            mark_value(v)

        # callframes: closures are Refs in this version; also super_ref
        for fr in vm.frames:
            mark_value(fr.closure_ref)
            mark_value(fr.super_ref) if fr.super_ref is not None else None

        # open upvalues: upvalue objects are heap objects; list of Refs
        for uvref in vm.open_upvalues.values():
            mark_value(uvref)

        # trace references (tri-color worklist). [page:1]
        while gray:
            oid = gray.pop()
            obj = self.objs.get(oid)
            if obj is None:
                continue
            for child in obj.children():
                mark_value(child)

        # sweep phase
        before = len(self.objs)
        dead = [oid for oid, o in self.objs.items() if not o.marked]
        for oid in dead:
            if self.trace_gc:
                o = self.objs[oid]
                print(f"[gc] sweep free oid={oid} kind={o.kind}")
            del self.objs[oid]

        # unmark survivors
        for o in self.objs.values():
            o.marked = False

        collected = len(dead)
        self.last_collected = collected

        # simple growth strategy (doubling) similar in spirit to nextGC growth. [page:1]
        self.next_gc = max(2000, len(self.objs) * 2 + 200)
        self.alloc_count = 0

        if self.trace_gc:
            after = len(self.objs)
            print(f"[gc] -- gc end -- collected={collected} alive={after} (was {before}) next_gc={self.next_gc}")

# =====================================
# Lexer
# =====================================

class Tok(Enum):
    EOF = auto()
    NEWLINE = auto()
    NUMBER = auto()
    STRING = auto()
    IDENT = auto()

    VAR = auto()
    CONST = auto()
    FUNCAO = auto()
    RETORNA = auto()

    CLASSE = auto()
    SUPER = auto()

    SE = auto()
    ENTAO = auto()
    SENAO = auto()
    ENQUANTO = auto()
    FACA = auto()
    FIM = auto()

    TENTA = auto()
    CAPTURA = auto()
    FINALMENTE = auto()
    LANCA = auto()

    VERDADE = auto()
    FALSO = auto()
    NIL = auto()

    ASSIGN = auto()
    EQ = auto()
    NE = auto()
    LT = auto()
    LE = auto()
    GT = auto()
    GE = auto()

    PLUS = auto()
    MINUS = auto()
    STAR = auto()
    SLASH = auto()
    POWER = auto()

    MOD = auto()
    AND = auto()
    OR = auto()
    NOT = auto()

    LPAREN = auto()
    RPAREN = auto()
    LBRACKET = auto()
    RBRACKET = auto()
    LBRACE = auto()
    RBRACE = auto()
    COMMA = auto()
    COLON = auto()
    DOT = auto()

@dataclass
class Token:
    t: Tok
    v: Any
    line: int
    col: int

class Lexer:
    def __init__(self, src: str):
        self.src = src
        self.i = 0
        self.line = 1
        self.col = 1
        self.keywords = {
            "var": Tok.VAR,
            "const": Tok.CONST,
            "funcao": Tok.FUNCAO,
            "retorna": Tok.RETORNA,

            "classe": Tok.CLASSE,
            "super": Tok.SUPER,

            "se": Tok.SE,
            "entao": Tok.ENTAO,
            "senao": Tok.SENAO,
            "enquanto": Tok.ENQUANTO,
            "faca": Tok.FACA,
            "fim": Tok.FIM,

            "tenta": Tok.TENTA,
            "captura": Tok.CAPTURA,
            "finalmente": Tok.FINALMENTE,
            "lanca": Tok.LANCA,

            "verdade": Tok.VERDADE,
            "falso": Tok.FALSO,
            "nil": Tok.NIL,

            "mod": Tok.MOD,
            "e": Tok.AND,
            "ou": Tok.OR,
            "nao": Tok.NOT,
        }

    def peek(self, k=0):
        j = self.i + k
        return self.src[j] if j < len(self.src) else None

    def adv(self):
        ch = self.peek()
        if ch is None:
            return None
        self.i += 1
        if ch == "
":
            self.line += 1
            self.col = 1
        else:
            self.col += 1
        return ch

    def skip_ws(self):
        while True:
            ch = self.peek()
            if ch is None:
                return
            if ch in " \t
":
                self.adv()
                continue
            if ch == "#":
                while self.peek() not in (None, "
"):
                    self.adv()
                continue
            return

    def read_number(self):
        line, col = self.line, self.col
        s = ""
        dot = False
        while True:
            ch = self.peek()
            if ch is None:
                break
            if ch.isdigit():
                s += self.adv()
            elif ch == "." and not dot:
                dot = True
                s += self.adv()
            else:
                break
        return Token(Tok.NUMBER, float(s) if dot else int(s), line, col)

    def read_ident(self):
        line, col = self.line, self.col
        s = ""
        while True:
            ch = self.peek()
            if ch is None:
                break
            if ch.isalnum() or ch == "_":
                s += self.adv()
            else:
                break
        tt = self.keywords.get(s, Tok.IDENT)
        if tt == Tok.VERDADE: return Token(tt, True, line, col)
        if tt == Tok.FALSO: return Token(tt, False, line, col)
        if tt == Tok.NIL: return Token(tt, None, line, col)
        return Token(tt, s, line, col)

    def read_string(self):
        quote = self.peek()
        line, col = self.line, self.col
        self.adv()
        out = ""
        while True:
            ch = self.peek()
            if ch is None:
                raise SyntaxError("String não terminada")
            if ch == quote:
                self.adv()
                break
            if ch == "\\":
                self.adv()
                esc = self.adv()
                if esc == "n": out += "
"
                elif esc == "t": out += "\t"
                elif esc == "r": out += "
"
                else: out += esc if esc is not None else ""
            else:
                out += self.adv()
        return Token(Tok.STRING, out, line, col)

    def next_token(self):
        self.skip_ws()
        ch = self.peek()
        if ch is None:
            return Token(Tok.EOF, None, self.line, self.col)
        if ch == "
" or ch == ";":
            line, col = self.line, self.col
            self.adv()
            return Token(Tok.NEWLINE, "
", line, col)
        if ch in ("'", '"'):
            return self.read_string()
        if ch.isdigit():
            return self.read_number()
        if ch.isalpha() or ch == "_":
            return self.read_ident()

        line, col = self.line, self.col

        if ch == "=":
            self.adv()
            if self.peek() == "=":
                self.adv()
                return Token(Tok.EQ, "==", line, col)
            return Token(Tok.ASSIGN, "=", line, col)
        if ch == "!":
            self.adv()
            if self.peek() == "=":
                self.adv()
                return Token(Tok.NE, "!=", line, col)
            raise SyntaxError("! inválido")
        if ch == "<":
            self.adv()
            if self.peek() == "=":
                self.adv()
                return Token(Tok.LE, "<=", line, col)
            return Token(Tok.LT, "<", line, col)
        if ch == ">":
            self.adv()
            if self.peek() == "=":
                self.adv()
                return Token(Tok.GE, ">=", line, col)
            return Token(Tok.GT, ">", line, col)

        if ch == "+":
            self.adv(); return Token(Tok.PLUS, "+", line, col)
        if ch == "-":
            self.adv(); return Token(Tok.MINUS, "-", line, col)
        if ch == "*":
            self.adv()
            if self.peek() == "*":
                self.adv()
                return Token(Tok.POWER, "**", line, col)
            return Token(Tok.STAR, "*", line, col)
        if ch == "/":
            self.adv(); return Token(Tok.SLASH, "/", line, col)

        if ch == "(":
            self.adv(); return Token(Tok.LPAREN, "(", line, col)
        if ch == ")":
            self.adv(); return Token(Tok.RPAREN, ")", line, col)
        if ch == "[":
            self.adv(); return Token(Tok.LBRACKET, "[", line, col)
        if ch == "]":
            self.adv(); return Token(Tok.RBRACKET, "]", line, col)
        if ch == "{":
            self.adv(); return Token(Tok.LBRACE, "{", line, col)
        if ch == "}":
            self.adv(); return Token(Tok.RBRACE, "}", line, col)
        if ch == ",":
            self.adv(); return Token(Tok.COMMA, ",", line, col)
        if ch == ":":
            self.adv(); return Token(Tok.COLON, ":", line, col)
        if ch == ".":
            self.adv(); return Token(Tok.DOT, ".", line, col)

        raise SyntaxError(f"Caractere inesperado: {ch!r} em {line}:{col}")

    def tokenize(self):
        out = []
        while True:
            t = self.next_token()
            out.append(t)
            if t.t == Tok.EOF:
                return out

# =====================================
# AST
# =====================================

class Node: pass
@dataclass
class NLiteral(Node): value: Any
@dataclass
class NIdent(Node): name: str
@dataclass
class NAssign(Node): target: Node; expr: Node; is_decl: bool
@dataclass
class NExprStmt(Node): expr: Node
@dataclass
class NCall(Node): fn: Node; args: List[Node]
@dataclass
class NBinary(Node): op: str; left: Node; right: Node
@dataclass
class NUnary(Node): op: str; expr: Node
@dataclass
class NIf(Node): cond: Node; then_body: List[Node]; else_body: Optional[List[Node]]
@dataclass
class NWhile(Node): cond: Node; body: List[Node]
@dataclass
class NFunc(Node): name: str; params: List[str]; body: List[Node]
@dataclass
class NReturn(Node): expr: Optional[Node]
@dataclass
class NList(Node): items: List[Node]
@dataclass
class NDict(Node): pairs: List[Tuple[Node, Node]]
@dataclass
class NIndex(Node): obj: Node; index: Node
@dataclass
class NAttr(Node): obj: Node; name: str
@dataclass
class NThrow(Node): expr: Node
@dataclass
class NTry(Node):
    try_body: List[Node]
    catch_name: Optional[str]
    catch_body: Optional[List[Node]]
    finally_body: Optional[List[Node]]
@dataclass
class NClass(Node):
    name: str
    superclass: Optional[str]
    methods: List[NFunc]
@dataclass
class NSuper(Node):
    name: str

# =====================================
# Parser (same as 2.1-D)
# =====================================

class Parser:
    def __init__(self, tokens: List[Token]):
        self.ts = tokens
        self.i = 0

    def cur(self) -> Token:
        return self.ts[self.i]

    def adv(self) -> Token:
        t = self.cur()
        if self.i < len(self.ts) - 1:
            self.i += 1
        return t

    def match(self, *kinds: Tok) -> Optional[Token]:
        if self.cur().t in kinds:
            return self.adv()
        return None

    def expect(self, kind: Tok):
        if self.cur().t != kind:
            t = self.cur()
            raise SyntaxError(f"Esperado {kind} em {t.line}:{t.col}, veio {t.t}")
        return self.adv()

    def skip_nl(self):
        while self.cur().t == Tok.NEWLINE:
            self.adv()

    def parse(self) -> List[Node]:
        out: List[Node] = []
        self.skip_nl()
        while self.cur().t != Tok.EOF:
            out.append(self.statement())
            self.skip_nl()
        return out

    def statement(self) -> Node:
        self.skip_nl()
        t = self.cur().t

        if t == Tok.CLASSE:
            return self.class_def()

        if t in (Tok.VAR, Tok.CONST):
            self.adv()
            name = self.expect(Tok.IDENT).v
            self.expect(Tok.ASSIGN)
            e = self.expr()
            return NAssign(NIdent(name), e, True)

        if t == Tok.FUNCAO:
            return self.func_def()

        if t == Tok.RETORNA:
            self.adv()
            if self.cur().t in (Tok.NEWLINE, Tok.EOF, Tok.FIM):
                return NReturn(None)
            return NReturn(self.expr())

        if t == Tok.SE:
            return self.if_stmt()

        if t == Tok.ENQUANTO:
            return self.while_stmt()

        if t == Tok.LANCA:
            self.adv()
            return NThrow(self.expr())

        if t == Tok.TENTA:
            return self.try_stmt()

        e = self.expr()
        return NExprStmt(e)

    def block_until(self, *enders: Tok) -> List[Node]:
        body: List[Node] = []
        self.skip_nl()
        while self.cur().t not in enders and self.cur().t != Tok.EOF:
            body.append(self.statement())
            self.skip_nl()
        return body

    def func_def(self) -> NFunc:
        self.expect(Tok.FUNCAO)
        name = self.expect(Tok.IDENT).v
        self.expect(Tok.LPAREN)
        params: List[str] = []
        if self.cur().t != Tok.RPAREN:
            while True:
                params.append(self.expect(Tok.IDENT).v)
                if not self.match(Tok.COMMA):
                    break
        self.expect(Tok.RPAREN)
        body = self.block_until(Tok.FIM)
        self.expect(Tok.FIM)
        return NFunc(name, params, body)

    def class_def(self) -> Node:
        self.expect(Tok.CLASSE)
        name = self.expect(Tok.IDENT).v
        superclass = None
        if self.match(Tok.LT):
            superclass = self.expect(Tok.IDENT).v
        methods: List[NFunc] = []
        self.skip_nl()
        while self.cur().t != Tok.FIM:
            if self.cur().t != Tok.FUNCAO:
                t = self.cur()
                raise SyntaxError(f"Em classe, esperado 'funcao', veio {t.t} em {t.line}:{t.col}")
            methods.append(self.func_def())
            self.skip_nl()
        self.expect(Tok.FIM)
        return NClass(name, superclass, methods)

    def if_stmt(self) -> Node:
        self.expect(Tok.SE)
        cond = self.expr()
        self.expect(Tok.ENTAO)
        then_body = self.block_until(Tok.SENAO, Tok.FIM)
        else_body = None
        if self.match(Tok.SENAO):
            else_body = self.block_until(Tok.FIM)
        self.expect(Tok.FIM)
        return NIf(cond, then_body, else_body)

    def while_stmt(self) -> Node:
        self.expect(Tok.ENQUANTO)
        cond = self.expr()
        self.expect(Tok.FACA)
        body = self.block_until(Tok.FIM)
        self.expect(Tok.FIM)
        return NWhile(cond, body)

    def try_stmt(self) -> Node:
        self.expect(Tok.TENTA)
        try_body = self.block_until(Tok.CAPTURA, Tok.FINALMENTE, Tok.FIM)

        catch_name = None
        catch_body = None
        finally_body = None

        if self.match(Tok.CAPTURA):
            if self.match(Tok.LPAREN):
                catch_name = self.expect(Tok.IDENT).v
                self.expect(Tok.RPAREN)
            else:
                catch_name = self.expect(Tok.IDENT).v
            catch_body = self.block_until(Tok.FINALMENTE, Tok.FIM)

        if self.match(Tok.FINALMENTE):
            finally_body = self.block_until(Tok.FIM)

        self.expect(Tok.FIM)
        if catch_body is None and finally_body is None:
            raise SyntaxError("tenta precisa de captura e/ou finalmente")
        return NTry(try_body, catch_name, catch_body, finally_body)

    def expr(self) -> Node:
        return self.or_expr()

    def or_expr(self):
        left = self.and_expr()
        while self.cur().t == Tok.OR:
            op = self.adv().v
            right = self.and_expr()
            left = NBinary(op, left, right)
        return left

    def and_expr(self):
        left = self.not_expr()
        while self.cur().t == Tok.AND:
            op = self.adv().v
            right = self.not_expr()
            left = NBinary(op, left, right)
        return left

    def not_expr(self):
        if self.cur().t == Tok.NOT:
            op = self.adv().v
            return NUnary(op, self.not_expr())
        return self.compare()

    def compare(self):
        left = self.add()
        while self.cur().t in (Tok.EQ, Tok.NE, Tok.LT, Tok.LE, Tok.GT, Tok.GE):
            op = self.adv().v
            right = self.add()
            left = NBinary(op, left, right)
        return left

    def add(self):
        left = self.mul()
        while self.cur().t in (Tok.PLUS, Tok.MINUS):
            op = self.adv().v
            right = self.mul()
            left = NBinary(op, left, right)
        return left

    def mul(self):
        left = self.power()
        while self.cur().t in (Tok.STAR, Tok.SLASH, Tok.MOD):
            op = self.adv().v
            right = self.power()
            left = NBinary(op, left, right)
        return left

    def power(self):
        left = self.unary()
        if self.cur().t == Tok.POWER:
            op = self.adv().v
            right = self.power()
            return NBinary(op, left, right)
        return left

    def unary(self):
        if self.cur().t == Tok.MINUS:
            op = self.adv().v
            return NUnary(op, self.unary())
        return self.postfix()

    def postfix(self):
        expr = self.primary()
        while True:
            if self.match(Tok.LPAREN):
                args: List[Node] = []
                if self.cur().t != Tok.RPAREN:
                    while True:
                        args.append(self.expr())
                        if not self.match(Tok.COMMA):
                            break
                self.expect(Tok.RPAREN)
                expr = NCall(expr, args)
                continue

            if self.match(Tok.LBRACKET):
                idx = self.expr()
                self.expect(Tok.RBRACKET)
                expr = NIndex(expr, idx)
                continue

            if self.match(Tok.DOT):
                name = self.expect(Tok.IDENT).v
                expr = NAttr(expr, name)
                continue

            if self.cur().t == Tok.ASSIGN and isinstance(expr, (NIdent, NIndex, NAttr)):
                self.adv()
                rhs = self.expr()
                expr = NAssign(expr, rhs, False)
                continue

            break
        return expr

    def primary(self):
        t = self.cur()
        if self.match(Tok.NUMBER): return NLiteral(t.v)
        if self.match(Tok.STRING): return NLiteral(t.v)
        if self.match(Tok.VERDADE) or self.match(Tok.FALSO) or self.match(Tok.NIL):
            return NLiteral(t.v)

        if self.match(Tok.SUPER):
            self.expect(Tok.DOT)
            name = self.expect(Tok.IDENT).v
            return NSuper(name)

        if self.match(Tok.IDENT): return NIdent(t.v)

        if self.match(Tok.LPAREN):
            e = self.expr()
            self.expect(Tok.RPAREN)
            return e

        if self.match(Tok.LBRACKET):
            items: List[Node] = []
            if self.cur().t != Tok.RBRACKET:
                while True:
                    items.append(self.expr())
                    if not self.match(Tok.COMMA):
                        break
            self.expect(Tok.RBRACKET)
            return NList(items)

        if self.match(Tok.LBRACE):
            pairs: List[Tuple[Node, Node]] = []
            if self.cur().t != Tok.RBRACE:
                while True:
                    k = self.expr()
                    self.expect(Tok.COLON)
                    v = self.expr()
                    pairs.append((k, v))
                    if not self.match(Tok.COMMA):
                        break
            self.expect(Tok.RBRACE)
            return NDict(pairs)

        raise SyntaxError(f"Token inesperado {t.t} em {t.line}:{t.col}")

# =====================================
# Bytecode structures
# =====================================

Instr = Tuple[Any, ...]

@dataclass
class Chunk:
    consts: List[Any]  # constants may include immediates and python strings; heap literals are created at runtime
    code: List[Instr]

@dataclass
class FunctionObj:
    name: str
    arity: int
    chunk: Chunk
    local_count: int
    upvalue_specs: List[Tuple[bool, int]]

@dataclass
class CallFrame:
    closure_ref: Ref          # Ref to ObjKind.CLOSURE
    ip: int
    base: int
    globals_ref: Dict[str, Any]
    super_ref: Optional[Ref]  # Ref to ObjKind.CLASS

@dataclass
class Handler:
    frame_index: int
    stack_depth: int
    catch_ip: Optional[int]
    finally_ip: Optional[int]
    end_ip: int
    catch_local_slot: Optional[int]
    state: int
    pending: Any

class CompileError(Exception): pass
class VMError(Exception): pass

# =====================================
# Compiler (mostly same as 2.1-D)
# Differences:
# - STRING/LIST/DICT literals are allocated at runtime via opcodes:
#   NEW_STRING constIndex, BUILD_LIST, BUILD_DICT
# - CLASS/METHOD/INHERIT produce heap objects
# =====================================

class SymbolTable:
    def __init__(self, parent: Optional["SymbolTable"]=None):
        self.parent = parent
        self.locals: Dict[str, int] = {}
        self.upvalues: Dict[str, int] = {}
    def define_local(self, name: str) -> int:
        if name in self.locals:
            return self.locals[name]
        idx = len(self.locals); self.locals[name] = idx; return idx
    def resolve_local(self, name: str) -> Optional[int]:
        return self.locals.get(name)

class Compiler:
    def __init__(self):
        self.current_consts: List[Any] = []
        self.current_code: List[Instr] = []
        self.sym_stack: List[SymbolTable] = []
        self.current_upvalue_specs: List[Tuple[bool, int]] = []

    def cur_sym(self) -> SymbolTable:
        return self.sym_stack[-1]

    def add_const(self, v: Any) -> int:
        self.current_consts.append(v); return len(self.current_consts) - 1

    def emit(self, *instr):
        self.current_code.append(tuple(instr))

    def emit_jump(self, op: str) -> int:
        self.emit(op, None); return len(self.current_code) - 1

    def patch_jump(self, at: int, target: Optional[int]=None):
        op, _ = self.current_code[at]
        self.current_code[at] = (op, len(self.current_code) if target is None else target)

    def add_upvalue(self, name: str, is_local: bool, index: int) -> int:
        cur = self.cur_sym()
        if name in cur.upvalues: return cur.upvalues[name]
        for i, (il, idx) in enumerate(self.current_upvalue_specs):
            if il == is_local and idx == index:
                cur.upvalues[name] = i; return i
        uv_index = len(self.current_upvalue_specs)
        cur.upvalues[name] = uv_index
        self.current_upvalue_specs.append((is_local, index))
        return uv_index

    def resolve_upvalue(self, name: str) -> Optional[int]:
        if len(self.sym_stack) < 2: return None
        enclosing = self.cur_sym().parent
        if enclosing is None: return None
        local = enclosing.resolve_local(name)
        if local is not None:
            return self.add_upvalue(name, True, local)
        cur_sym = self.sym_stack.pop()
        cur_specs = self.current_upvalue_specs
        up_in_enclosing = self.resolve_upvalue(name)
        self.sym_stack.append(cur_sym)
        self.current_upvalue_specs = cur_specs
        if up_in_enclosing is not None:
            return self.add_upvalue(name, False, up_in_enclosing)
        return None

    def compile_program(self, nodes: List[Node]) -> FunctionObj:
        self.current_consts, self.current_code = [], []
        self.sym_stack = [SymbolTable(None)]
        self.current_upvalue_specs = []
        for n in nodes:
            self.compile_stmt(n)
        k = self.add_const(None)
        self.emit("CONST", k)
        self.emit("RETURN")
        return FunctionObj("<script>", 0, Chunk(self.current_consts, self.current_code),
                           local_count=len(self.cur_sym().locals),
                           upvalue_specs=[])

    def compile_stmt(self, n: Node):
        if isinstance(n, NClass):
            self.compile_class(n); return
        if isinstance(n, NFunc):
            self.compile_function_def(n); return
        if isinstance(n, NReturn):
            if n.expr is None:
                k = self.add_const(None); self.emit("CONST", k)
            else:
                self.compile_expr(n.expr)
            self.emit("RETURN"); return
        if isinstance(n, NThrow):
            self.compile_expr(n.expr); self.emit("THROW"); return
        if isinstance(n, NTry):
            self.compile_try(n); return
        if isinstance(n, NIf):
            self.compile_expr(n.cond)
            jmp_false = self.emit_jump("JMP_IF_FALSE")
            self.emit("POP")
            for s in n.then_body: self.compile_stmt(s)
            if n.else_body is not None:
                jmp_end = self.emit_jump("JMP")
                self.patch_jump(jmp_false); self.emit("POP")
                for s in n.else_body: self.compile_stmt(s)
                self.patch_jump(jmp_end)
            else:
                self.patch_jump(jmp_false); self.emit("POP")
            return
        if isinstance(n, NWhile):
            loop_start = len(self.current_code)
            self.compile_expr(n.cond)
            jmp_false = self.emit_jump("JMP_IF_FALSE")
            self.emit("POP")
            for s in n.body: self.compile_stmt(s)
            self.emit("JMP", loop_start)
            self.patch_jump(jmp_false); self.emit("POP")
            return
        if isinstance(n, NExprStmt):
            self.compile_expr(n.expr); self.emit("POP"); return
        if isinstance(n, NAssign) and n.is_decl:
            self.compile_expr(n.expr)
            if not isinstance(n.target, NIdent):
                raise CompileError("Declaração var só pode ser em identificador")
            idx = self.cur_sym().define_local(n.target.name)
            self.emit("STORE_LOCAL", idx); return
        raise CompileError(f"Stmt não suportado: {type(n).__name__}")

    def compile_try(self, n: NTry):
        enter_at = len(self.current_code)
        self.emit("TRY_ENTER", None, None, None, None)
        for s in n.try_body: self.compile_stmt(s)
        jmp_to_finally_or_end = self.emit_jump("JMP")

        catch_ip = None
        catch_slot = None
        if n.catch_body is not None:
            catch_ip = len(self.current_code)
            if not n.catch_name:
                raise CompileError("captura precisa de nome")
            catch_slot = self.cur_sym().define_local(n.catch_name)
            self.emit("STORE_LOCAL", catch_slot)
            for s in n.catch_body: self.compile_stmt(s)
            self.emit("JMP", None)

        finally_ip = None
        if n.finally_body is not None:
            finally_ip = len(self.current_code)
            for s in n.finally_body: self.compile_stmt(s)

        end_ip = len(self.current_code)
        self.emit("TRY_EXIT")

        self.patch_jump(jmp_to_finally_or_end, finally_ip if finally_ip is not None else end_ip)
        if catch_ip is not None:
            for i in range(catch_ip, end_ip):
                instr = self.current_code[i]
                if instr[0] == "JMP" and instr[1] is None:
                    self.current_code[i] = ("JMP", finally_ip if finally_ip is not None else end_ip)

        self.current_code[enter_at] = ("TRY_ENTER", catch_ip, finally_ip, end_ip, catch_slot)

    def compile_function_def(self, n: NFunc):
        saved_consts, saved_code = self.current_consts, self.current_code
        saved_stack = self.sym_stack
        saved_specs = self.current_upvalue_specs

        self.current_consts, self.current_code = [], []
        enclosing = self.cur_sym() if self.sym_stack else None
        self.sym_stack = saved_stack + [SymbolTable(enclosing)]
        self.current_upvalue_specs = []

        for p in n.params:
            self.cur_sym().define_local(p)
        for s in n.body:
            self.compile_stmt(s)

        k = self.add_const(None)
        self.emit("CONST", k)
        self.emit("RETURN")

        fn_obj = FunctionObj(n.name, len(n.params), Chunk(self.current_consts, self.current_code),
                             local_count=len(self.cur_sym().locals),
                             upvalue_specs=self.current_upvalue_specs.copy())

        self.sym_stack = saved_stack
        self.current_consts, self.current_code = saved_consts, saved_code
        self.current_upvalue_specs = saved_specs

        kf = self.add_const(fn_obj)
        self.emit("MAKE_CLOSURE", kf)  # alloc closure in heap
        slot = self.cur_sym().define_local(n.name)
        self.emit("STORE_LOCAL", slot)

    def compile_class(self, n: NClass):
        kname = self.add_const(n.name)
        self.emit("CLASS", kname)

        slot = self.cur_sym().define_local(n.name)
        self.emit("DUP")
        self.emit("STORE_LOCAL", slot)

        if n.superclass is not None:
            self.emit("LOAD_GLOBAL", n.superclass)
            self.emit("INHERIT")

        for m in n.methods:
            self.compile_method(m)

        self.emit("POP")

    def compile_method(self, m: NFunc):
        saved_consts, saved_code = self.current_consts, self.current_code
        saved_stack = self.sym_stack
        saved_specs = self.current_upvalue_specs

        self.current_consts, self.current_code = [], []
        enclosing = self.cur_sym() if self.sym_stack else None
        self.sym_stack = saved_stack + [SymbolTable(enclosing)]
        self.current_upvalue_specs = []

        for p in m.params:
            self.cur_sym().define_local(p)
        for s in m.body:
            self.compile_stmt(s)

        k = self.add_const(None)
        self.emit("CONST", k)
        self.emit("RETURN")

        fn_obj = FunctionObj(m.name, len(m.params), Chunk(self.current_consts, self.current_code),
                             local_count=len(self.cur_sym().locals),
                             upvalue_specs=self.current_upvalue_specs.copy())

        self.sym_stack = saved_stack
        self.current_consts, self.current_code = saved_consts, saved_code
        self.current_upvalue_specs = saved_specs

        kfn = self.add_const(fn_obj)
        self.emit("MAKE_CLOSURE", kfn)
        km = self.add_const(m.name)
        self.emit("METHOD", km)

    def compile_expr(self, n: Node):
        if isinstance(n, NLiteral):
            # strings are heap allocated now (NEW_STRING); others can be CONST immediates
            if isinstance(n.value, str):
                k = self.add_const(n.value)
                self.emit("NEW_STRING", k)
            else:
                k = self.add_const(n.value)
                self.emit("CONST", k)
            return

        if isinstance(n, NList):
            for it in n.items: self.compile_expr(it)
            self.emit("BUILD_LIST", len(n.items))
            return

        if isinstance(n, NDict):
            for k_node, v_node in n.pairs:
                self.compile_expr(k_node); self.compile_expr(v_node)
            self.emit("BUILD_DICT", len(n.pairs))
            return

        if isinstance(n, NIdent):
            local = self.cur_sym().resolve_local(n.name)
            if local is not None:
                self.emit("LOAD_LOCAL", local); return
            uv = self.resolve_upvalue(n.name)
            if uv is not None:
                self.emit("GET_UPVALUE", uv); return
            self.emit("LOAD_GLOBAL", n.name); return

        if isinstance(n, NSuper):
            k = self.add_const(n.name)
            self.emit("GET_SUPER", k)
            return

        if isinstance(n, NUnary):
            self.compile_expr(n.expr)
            if n.op == "-": self.emit("NEG")
            elif n.op == "nao": self.emit("NOT")
            else: raise CompileError(f"Unary op desconhecido: {n.op}")
            return

        if isinstance(n, NBinary):
            if n.op == "e":
                self.compile_expr(n.left)
                jmp = self.emit_jump("JMP_IF_FALSE_KEEP")
                self.emit("POP")
                self.compile_expr(n.right)
                self.patch_jump(jmp)
                return
            if n.op == "ou":
                self.compile_expr(n.left)
                jmp = self.emit_jump("JMP_IF_TRUE_KEEP")
                self.emit("POP")
                self.compile_expr(n.right)
                self.patch_jump(jmp)
                return

            self.compile_expr(n.left); self.compile_expr(n.right)
            op = n.op
            if op == "+": self.emit("ADD")
            elif op == "-": self.emit("SUB")
            elif op == "*": self.emit("MUL")
            elif op == "/": self.emit("DIV")
            elif op == "**": self.emit("POW")
            elif op == "mod": self.emit("MOD")
            elif op == "==": self.emit("EQ")
            elif op == "!=": self.emit("NE")
            elif op == "<": self.emit("LT")
            elif op == "<=": self.emit("LE")
            elif op == ">": self.emit("GT")
            elif op == ">=": self.emit("GE")
            else: raise CompileError(f"Binary op desconhecido: {op}")
            return

        if isinstance(n, NIndex):
            self.compile_expr(n.obj); self.compile_expr(n.index)
            self.emit("GET_INDEX"); return

        if isinstance(n, NAttr):
            self.compile_expr(n.obj)
            k = self.add_const(n.name)
            self.emit("GET_PROPERTY", k)
            return

        if isinstance(n, NCall):
            self.compile_expr(n.fn)
            for a in n.args: self.compile_expr(a)
            self.emit("CALL_VALUE", len(n.args)); return

        if isinstance(n, NAssign) and not n.is_decl:
            self.compile_expr(n.expr)
            if isinstance(n.target, NIdent):
                local = self.cur_sym().resolve_local(n.target.name)
                if local is not None:
                    self.emit("DUP"); self.emit("STORE_LOCAL", local); return
                uv = self.resolve_upvalue(n.target.name)
                if uv is not None:
                    self.emit("DUP"); self.emit("SET_UPVALUE", uv); return
                raise CompileError(f"Atribuição global não suportada: {n.target.name}")

            if isinstance(n.target, NIndex):
                tmp = self.cur_sym().define_local("__tmp_assign")
                self.emit("STORE_LOCAL", tmp)
                self.compile_expr(n.target.obj)
                self.compile_expr(n.target.index)
                self.emit("LOAD_LOCAL", tmp)
                self.emit("SET_INDEX")
                return

            if isinstance(n.target, NAttr):
                tmp = self.cur_sym().define_local("__tmp_assign")
                self.emit("STORE_LOCAL", tmp)
                self.compile_expr(n.target.obj)
                self.emit("LOAD_LOCAL", tmp)
                kname = self.add_const(n.target.name)
                self.emit("SET_PROPERTY", kname)
                return

            raise CompileError("Destino de atribuição inválido")

        raise CompileError(f"Expr não suportada: {type(n).__name__}")

# =====================================
# VM
# =====================================

class VM:
    def __init__(self, entry_file: str, trace_gc: bool = False):
        self.heap = Arena()
        self.heap.trace_gc = trace_gc

        self.stack: List[Any] = []
        self.frames: List[CallFrame] = []
        self.handlers: List[Handler] = []

        # open upvalues: map from stack slot index -> Ref(UPVALUE)
        self.open_upvalues: Dict[int, Ref] = {}

        self.modules: Dict[str, Dict[str, Any]] = {}
        self.entry_file = os.path.abspath(entry_file)
        self.entry_dir = os.path.dirname(self.entry_file)

        self.builtins: Dict[str, Any] = {}
        self._init_builtins()

        self.root_globals: Dict[str, Any] = {}
        self.root_globals.update(self.builtins)

    # ---- value helpers
    def is_ref(self, v: Any) -> bool:
        return isinstance(v, Ref)

    def deref(self, r: Ref) -> Obj:
        return self.heap.get(r)

    def to_py_str(self, v: Any) -> str:
        if isinstance(v, Ref) and self.deref(v).kind == ObjKind.STRING:
            return self.deref(v).payload["s"]
        return str(v)

    def truthy(self, v: Any) -> bool:
        if v is None: return False
        if isinstance(v, bool): return v
        return bool(v)

    # ---- heap alloc helpers
    def new_string(self, s: str) -> Ref:
        r = self.heap.alloc(ObjKind.STRING, {"s": s})
        self.heap.maybe_collect(self)
        return r

    def new_list(self, items: List[Any]) -> Ref:
        r = self.heap.alloc(ObjKind.LIST, {"items": items})
        self.heap.maybe_collect(self)
        return r

    def new_dict(self, items: Dict[Any, Any]) -> Ref:
        r = self.heap.alloc(ObjKind.DICT, {"items": items})
        self.heap.maybe_collect(self)
        return r

    def new_upvalue(self, is_open: bool, slot: Optional[int], closed: Any) -> Ref:
        r = self.heap.alloc(ObjKind.UPVALUE, {"is_open": is_open, "slot": slot, "closed": closed})
        self.heap.maybe_collect(self)
        return r

    def new_closure(self, fn: FunctionObj, upvalues: List[Ref]) -> Ref:
        # function itself is stored as a python object; wrap it in heap too so GC can traverse constants if needed
        fn_ref = self.heap.alloc(ObjKind.DICT, {"items": {}})  # placeholder node to keep function reachable
        self.heap.get(fn_ref).kind = ObjKind.DICT
        self.heap.get(fn_ref).payload = {"items": {"__function__": fn}}
        r = self.heap.alloc(ObjKind.CLOSURE, {"function_ref": fn_ref, "upvalues": upvalues})
        self.heap.maybe_collect(self)
        return r

    def closure_fn(self, clo_ref: Ref) -> FunctionObj:
        clo = self.deref(clo_ref)
        fn_ref = clo.payload["function_ref"]
        fn_box = self.deref(fn_ref).payload["items"]["__function__"]
        return fn_box

    def closure_upvalues(self, clo_ref: Ref) -> List[Ref]:
        return self.deref(clo_ref).payload["upvalues"]

    def new_class(self, name: str) -> Ref:
        r = self.heap.alloc(ObjKind.CLASS, {"name": name, "methods": {}, "superclass": None})
        self.heap.maybe_collect(self)
        return r

    def new_instance(self, klass_ref: Ref) -> Ref:
        r = self.heap.alloc(ObjKind.INSTANCE, {"klass": klass_ref, "fields": {}})
        self.heap.maybe_collect(self)
        return r

    def new_bound_method(self, receiver: Ref, method: Ref) -> Ref:
        r = self.heap.alloc(ObjKind.BOUND_METHOD, {"receiver": receiver, "method": method})
        self.heap.maybe_collect(self)
        return r

    # ---- builtins / modules
    def _init_builtins(self):
        self.builtins["escreva"] = lambda *args: (print(*[self.to_py_str(a) for a in args]), None)[1]
        self.builtins["tamanho"] = lambda x: self._len(x)
        self.builtins["tipo"] = lambda x: self._type_name(x)

        self.builtins["gc"] = {
            "coleta": lambda: (self.heap.collect(self), None)[1],
            "stats": lambda: {"objs": len(self.heap.objs), "next_gc": self.heap.next_gc, "last_collected": self.heap.last_collected},
        }

        fs = {
            "leia_texto": lambda path: open(path, "r", encoding="utf-8").read(),
            "escreva_texto": lambda path, conteudo: (open(path, "w", encoding="utf-8").write(self.to_py_str(conteudo)), None)[1],
            "existe": lambda path: os.path.exists(path),
            "lista_dir": lambda path=".": os.listdir(path),
        }
        js = {
            "parse": lambda s: py_json.loads(s),
            "stringify": lambda obj, bonito=False: py_json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) if bonito else py_json.dumps(obj, ensure_ascii=False),
        }
        tm = {
            "agora_ms": lambda: int(py_time.time() * 1000),
            "espera_ms": lambda ms: (py_time.sleep(ms / 1000.0), None)[1],
        }
        args = {"lista": sys.argv[1:]}
        env = {"get": lambda nome, padrao=None: os.environ.get(nome, padrao)}

        self.builtins["fs"] = fs
        self.builtins["json"] = js
        self.builtins["time"] = tm
        self.builtins["args"] = args
        self.builtins["env"] = env

        self.builtins["importa"] = self._builtin_importa

    def _len(self, x: Any) -> int:
        if isinstance(x, Ref):
            o = self.deref(x)
            if o.kind == ObjKind.STRING:
                return len(o.payload["s"])
            if o.kind == ObjKind.LIST:
                return len(o.payload["items"])
            if o.kind == ObjKind.DICT:
                return len(o.payload["items"])
        return len(x)

    def _type_name(self, x: Any) -> str:
        if isinstance(x, Ref):
            return f"spyro_{self.deref(x).kind.value}"
        return type(x).__name__

    def _search_paths(self) -> List[str]:
        paths = [self.entry_dir]
        spyro_path = os.environ.get("SPYRO_PATH")
        if spyro_path:
            sep = ";" if os.name == "nt" else ":"
            for p in spyro_path.split(sep):
                p = p.strip()
                if p:
                    paths.append(os.path.abspath(p))
        stdlib = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stdlib")
        paths.append(stdlib)
        out = []
        seen = set()
        for p in paths:
            if p not in seen:
                out.append(p); seen.add(p)
        return out

    def _resolve_module(self, name_or_path: str) -> str:
        cand_names = []
        raw = name_or_path
        if raw.endswith(".spy"):
            cand_names.append(raw)
        else:
            cand_names.append(raw + ".spy")
            cand_names.append(raw)

        for nm in cand_names:
            if os.path.isabs(nm) and os.path.exists(nm):
                return os.path.abspath(nm)

        for base in self._search_paths():
            for nm in cand_names:
                p = os.path.abspath(os.path.join(base, nm))
                if os.path.exists(p):
                    return p

        raise VMError(f"importa(): módulo não encontrado: {name_or_path}")

    def _builtin_importa(self, name_or_path: str):
        path = self._resolve_module(str(name_or_path))
        if path in self.modules:
            return self.modules[path]

        mod_globals: Dict[str, Any] = {}
        mod_globals.update(self.builtins)
        self.modules[path] = mod_globals

        src = open(path, "r", encoding="utf-8").read()
        toks = Lexer(src).tokenize()
        ast = Parser(toks).parse()

        comp = Compiler()
        fn = comp.compile_program(ast)
        entry_clo = self.new_closure(fn, [])

        self._run_closure(entry_clo, mod_globals, super_ref=None)
        return mod_globals

    # ---- stack helpers
    def push(self, v: Any): self.stack.append(v)
    def pop(self) -> Any: return self.stack.pop()
    def peek(self) -> Any: return self.stack[-1]

    # ---- upvalues
    def capture_upvalue(self, stack_slot: int) -> Ref:
        if stack_slot in self.open_upvalues:
            return self.open_upvalues[stack_slot]
        uv = self.new_upvalue(is_open=True, slot=stack_slot, closed=None)
        self.open_upvalues[stack_slot] = uv
        return uv

    def close_upvalues_from(self, base_slot: int):
        to_close = [k for k in self.open_upvalues.keys() if k >= base_slot]
        for k in sorted(to_close, reverse=True):
            uvref = self.open_upvalues.pop(k)
            uv = self.deref(uvref)
            uv.payload["closed"] = self.stack[k]
            uv.payload["is_open"] = False
            uv.payload["slot"] = None

    def upvalue_get(self, uvref: Ref) -> Any:
        uv = self.deref(uvref).payload
        if uv["is_open"]:
            return self.stack[uv["slot"]]
        return uv["closed"]

    def upvalue_set(self, uvref: Ref, v: Any):
        uv = self.deref(uvref).payload
        if uv["is_open"]:
            self.stack[uv["slot"]] = v
        else:
            uv["closed"] = v

    # ---- containers (Spyro heap)
    def get_index(self, obj: Any, idx: Any):
        if isinstance(obj, Ref):
            o = self.deref(obj)
            if o.kind == ObjKind.LIST:
                return o.payload["items"][idx]
            if o.kind == ObjKind.DICT:
                return o.payload["items"][idx]
        return obj[idx]

    def set_index(self, obj: Any, idx: Any, value: Any):
        if isinstance(obj, Ref):
            o = self.deref(obj)
            if o.kind == ObjKind.LIST:
                o.payload["items"][idx] = value
                return value
            if o.kind == ObjKind.DICT:
                o.payload["items"][idx] = value
                return value
        obj[idx] = value
        return value

    # ---- OO
    def class_name(self, klass_ref: Ref) -> str:
        return self.deref(klass_ref).payload["name"]

    def find_method(self, klass_ref: Ref, name: str) -> Optional[Ref]:
        k = self.deref(klass_ref).payload
        if name in k["methods"]:
            return k["methods"][name]
        if k["superclass"] is not None:
            return self.find_method(k["superclass"], name)
        return None

    def get_property(self, obj: Any, name: str):
        if isinstance(obj, Ref):
            o = self.deref(obj)
            if o.kind == ObjKind.INSTANCE:
                fields = o.payload["fields"]
                if name in fields:
                    return fields[name]
                m = self.find_method(o.payload["klass"], name)
                if m is not None:
                    return self.new_bound_method(obj, m)
                raise VMError(f"Propriedade indefinida: {self.class_name(o.payload['klass'])}.{name}")
            if o.kind == ObjKind.DICT:
                items = o.payload["items"]
                if name in items:
                    return items[name]
                raise VMError(f"Atributo não encontrado: .{name}")
            if o.kind == ObjKind.LIST and name == "tamanho":
                return len(o.payload["items"])
            if o.kind == ObjKind.STRING and name == "tamanho":
                return len(o.payload["s"])
        raise VMError(f"Tipo não suporta propriedade: {type(obj).__name__}.{name}")

    def set_property(self, obj: Any, name: str, value: Any):
        if isinstance(obj, Ref):
            o = self.deref(obj)
            if o.kind == ObjKind.INSTANCE:
                o.payload["fields"][name] = value
                return value
            if o.kind == ObjKind.DICT:
                o.payload["items"][name] = value
                return value
        raise VMError(f"Tipo não suporta set propriedade: {type(obj).__name__}.{name}")

    # ---- exceptions
    def _unwind_to(self, frame_index: int, stack_depth: int):
        while len(self.frames) - 1 > frame_index:
            fr = self.frames.pop()
            self.close_upvalues_from(fr.base)
            while len(self.stack) > fr.base:
                self.pop()
        while len(self.stack) > stack_depth:
            self.pop()

    def _throw(self, exc_value: Any):
        while self.handlers:
            h = self.handlers[-1]
            if h.frame_index >= len(self.frames):
                self.handlers.pop()
                continue

            if h.finally_ip is not None and h.state != 2:
                self._unwind_to(h.frame_index, h.stack_depth)
                h.pending = ("throw", exc_value)
                h.state = 2
                self.frames[h.frame_index].ip = h.finally_ip
                return

            if h.catch_ip is not None and h.state == 0:
                self._unwind_to(h.frame_index, h.stack_depth)
                self.push(exc_value)
                h.state = 1
                self.frames[h.frame_index].ip = h.catch_ip
                return

            self.handlers.pop()

        # Unhandled: include stack trace for debugging. [page:1]
        st = [self.closure_fn(fr.closure_ref).name for fr in self.frames]
        raise VMError(f"Exceção não tratada: {exc_value!r}
Call stack: " + " -> ".join(st))

    # ---- calls
    def call_closure(self, clo_ref: Ref, args: List[Any], globals_ref: Dict[str, Any], super_ref: Optional[Ref]):
        fn = self.closure_fn(clo_ref)
        if len(args) != fn.arity:
            raise VMError(f"{fn.name}: esperado {fn.arity} args, recebeu {len(args)}")
        base = len(self.stack)
        self.frames.append(CallFrame(closure_ref=clo_ref, ip=0, base=base, globals_ref=globals_ref, super_ref=super_ref))
        for a in args: self.push(a)
        for _ in range(fn.local_count - fn.arity):
            self.push(None)

    def call_value(self, callee: Any, args: List[Any], globals_ref: Dict[str, Any], super_ref: Optional[Ref]):
        if isinstance(callee, Ref):
            o = self.deref(callee)
            if o.kind == ObjKind.BOUND_METHOD:
                recv = o.payload["receiver"]
                method = o.payload["method"]
                # super_ref for method body is the superclass of receiver's class
                recv_class = self.deref(recv).payload["klass"]
                sup = self.deref(recv_class).payload["superclass"]
                self.call_closure(method, [recv] + args, globals_ref, sup)
                return "pending"

            if o.kind == ObjKind.CLASS:
                inst = self.new_instance(callee)
                init = self.find_method(callee, "__init__")
                if init is not None:
                    self.call_closure(init, [inst] + args, globals_ref, self.deref(callee).payload["superclass"])
                    # __init__ return ignored; we force instance on RETURN
                    return "pending_init"
                self.push(inst)
                return None

            if o.kind == ObjKind.CLOSURE:
                self.call_closure(callee, args, globals_ref, super_ref)
                return "pending"

        if callable(callee):
            ret = callee(*args)
            self.push(ret)
            return None

        raise VMError("Tentou chamar algo que não é função.")

    # ---- execute closure
    def _run_closure(self, entry_clo_ref: Ref, globals_dict: Dict[str, Any], super_ref: Optional[Ref]):
        # push initial frame
        self.frames.append(CallFrame(entry_clo_ref, ip=0, base=len(self.stack), globals_ref=globals_dict, super_ref=super_ref))
        fn = self.closure_fn(entry_clo_ref)
        for _ in range(fn.local_count):
            self.push(None)

        while True:
            frame = self.frames[-1]
            clo_ref = frame.closure_ref
            fn = self.closure_fn(clo_ref)
            code = fn.chunk.code
            consts = fn.chunk.consts

            instr = code[frame.ip]
            frame.ip += 1
            op = instr[0]

            try:
                if op == "CONST":
                    self.push(consts[instr[1]]); continue
                if op == "NEW_STRING":
                    s = consts[instr[1]]
                    self.push(self.new_string(s)); continue
                if op == "POP":
                    self.pop(); continue
                if op == "DUP":
                    self.push(self.peek()); continue

                if op == "NEG":
                    self.push(-self.pop()); continue
                if op == "NOT":
                    self.push(not self.truthy(self.pop())); continue

                if op in ("ADD","SUB","MUL","DIV","POW","MOD","EQ","NE","LT","LE","GT","GE"):
                    b = self.pop(); a = self.pop()
                    if op == "ADD":
                        # string concat if any operand is string
                        if isinstance(a, Ref) and self.deref(a).kind == ObjKind.STRING:
                            self.push(self.new_string(self.to_py_str(a) + self.to_py_str(b)))
                        elif isinstance(b, Ref) and self.deref(b).kind == ObjKind.STRING:
                            self.push(self.new_string(self.to_py_str(a) + self.to_py_str(b)))
                        else:
                            self.push(a + b)
                    elif op == "SUB": self.push(a - b)
                    elif op == "MUL": self.push(a * b)
                    elif op == "DIV": self.push(a / b)
                    elif op == "POW": self.push(a ** b)
                    elif op == "MOD": self.push(a % b)
                    elif op == "EQ": self.push(a == b)
                    elif op == "NE": self.push(a != b)
                    elif op == "LT": self.push(a < b)
                    elif op == "LE": self.push(a <= b)
                    elif op == "GT": self.push(a > b)
                    elif op == "GE": self.push(a >= b)
                    continue

                if op == "JMP":
                    frame.ip = instr[1]; continue
                if op == "JMP_IF_FALSE":
                    if not self.truthy(self.peek()): frame.ip = instr[1]
                    continue
                if op == "JMP_IF_FALSE_KEEP":
                    if not self.truthy(self.peek()): frame.ip = instr[1]
                    continue
                if op == "JMP_IF_TRUE_KEEP":
                    if self.truthy(self.peek()): frame.ip = instr[1]
                    continue

                if op == "LOAD_LOCAL":
                    self.push(self.stack[frame.base + instr[1]]); continue
                if op == "STORE_LOCAL":
                    idx = instr[1]; v = self.pop()
                    self.stack[frame.base + idx] = v
                    continue

                if op == "LOAD_GLOBAL":
                    name = instr[1]
                    if name in frame.globals_ref:
                        self.push(frame.globals_ref[name])
                    else:
                        raise VMError(f"Global/builtin indefinido: {name}")
                    continue

                # closures in heap
                if op == "MAKE_CLOSURE":
                    fn_obj = consts[instr[1]]
                    upvalues: List[Ref] = []
                    for is_local, index in fn_obj.upvalue_specs:
                        if is_local:
                            upvalues.append(self.capture_upvalue(frame.base + index))
                        else:
                            upvalues.append(self.closure_upvalues(clo_ref)[index])
                    self.push(self.new_closure(fn_obj, upvalues))
                    continue

                if op == "GET_UPVALUE":
                    uvref = self.closure_upvalues(clo_ref)[instr[1]]
                    self.push(self.upvalue_get(uvref)); continue
                if op == "SET_UPVALUE":
                    uvref = self.closure_upvalues(clo_ref)[instr[1]]
                    v = self.pop(); self.upvalue_set(uvref, v); self.push(v); continue

                # heap containers
                if op == "BUILD_LIST":
                    n = instr[1]
                    items = [self.pop() for _ in range(n)][::-1]
                    self.push(self.new_list(items))
                    continue

                if op == "BUILD_DICT":
                    n = instr[1]
                    d = {}
                    for _ in range(n):
                        v = self.pop(); k = self.pop()
                        d[k] = v
                    self.push(self.new_dict(d))
                    continue

                if op == "GET_INDEX":
                    idx = self.pop(); obj = self.pop()
                    self.push(self.get_index(obj, idx)); continue
                if op == "SET_INDEX":
                    val = self.pop(); idx = self.pop(); obj = self.pop()
                    self.push(self.set_index(obj, idx, val)); continue

                # exceptions
                if op == "TRY_ENTER":
                    catch_ip, finally_ip, end_ip, catch_slot = instr[1], instr[2], instr[3], instr[4]
                    self.handlers.append(Handler(
                        frame_index=len(self.frames) - 1,
                        stack_depth=len(self.stack),
                        catch_ip=catch_ip,
                        finally_ip=finally_ip,
                        end_ip=end_ip,
                        catch_local_slot=catch_slot,
                        state=0,
                        pending=None
                    ))
                    continue
                if op == "TRY_EXIT":
                    if self.handlers and self.handlers[-1].frame_index == len(self.frames) - 1:
                        h = self.handlers[-1]
                        if h.finally_ip is not None and h.state != 2:
                            h.pending = None
                            h.state = 2
                            frame.ip = h.finally_ip
                            continue
                        self.handlers.pop()
                    continue
                if op == "THROW":
                    self._throw(self.pop()); continue

                # OO (heap)
                if op == "CLASS":
                    name = consts[instr[1]]
                    self.push(self.new_class(name)); continue
                if op == "INHERIT":
                    superclass = self.pop()
                    klass = self.peek()
                    if not (isinstance(superclass, Ref) and isinstance(klass, Ref)):
                        raise VMError("INHERIT espera refs")
                    if self.deref(superclass).kind != ObjKind.CLASS or self.deref(klass).kind != ObjKind.CLASS:
                        raise VMError("INHERIT espera CLASS refs")
                    self.deref(klass).payload["superclass"] = superclass
                    continue
                if op == "METHOD":
                    mname = consts[instr[1]]
                    method = self.pop()
                    klass = self.peek()
                    if self.deref(klass).kind != ObjKind.CLASS:
                        raise VMError("METHOD espera CLASS no topo")
                    if not (isinstance(method, Ref) and self.deref(method).kind == ObjKind.CLOSURE):
                        raise VMError("METHOD espera CLOSURE")
                    self.deref(klass).payload["methods"][mname] = method
                    continue
                if op == "GET_PROPERTY":
                    name = consts[instr[1]]
                    obj = self.pop()
                    self.push(self.get_property(obj, name))
                    continue
                if op == "SET_PROPERTY":
                    name = consts[instr[1]]
                    val = self.pop()
                    obj = self.pop()
                    self.push(self.set_property(obj, name, val))
                    continue
                if op == "GET_SUPER":
                    name = consts[instr[1]]
                    receiver = self.peek()
                    if not (isinstance(receiver, Ref) and self.deref(receiver).kind == ObjKind.INSTANCE):
                        raise VMError("super usado sem self instância")
                    if frame.super_ref is None:
                        raise VMError("super usado em classe sem superclass")
                    m = self.find_method(frame.super_ref, name)
                    if m is None:
                        raise VMError(f"Método super indefinido: {self.class_name(frame.super_ref)}.{name}")
                    self.push(self.new_bound_method(receiver, m))
                    continue

                # calls/returns
                if op == "CALL_VALUE":
                    argc = instr[1]
                    args = [self.pop() for _ in range(argc)][::-1]
                    callee = self.pop()
                    res = self.call_value(callee, args, frame.globals_ref, frame.super_ref)
                    if res is None:
                        continue
                    continue

                if op == "RETURN":
                    ret = self.pop()

                    # __init__ forces returning receiver instance (self at local 0)
                    if fn.name == "__init__":
                        ret = self.stack[frame.base + 0]

                    # handle pending finally on return
                    while self.handlers:
                        h = self.handlers[-1]
                        if h.frame_index != len(self.frames) - 1:
                            break
                        if h.finally_ip is not None and h.state != 2:
                            self._unwind_to(h.frame_index, h.stack_depth)
                            h.pending = ("return", ret)
                            h.state = 2
                            frame.ip = h.finally_ip
                            ret = None
                            break
                        else:
                            self.handlers.pop()
                    if ret is None and self.handlers and self.handlers[-1].pending is not None and self.handlers[-1].state == 2:
                        continue

                    # pop frame
                    self.close_upvalues_from(frame.base)
                    base = frame.base
                    while len(self.stack) > base:
                        self.pop()
                    self.frames.pop()
                    if not self.frames:
                        return ret
                    self.push(ret)
                    continue

                raise VMError(f"Opcode desconhecido: {op}")

            except VMError as e:
                self._throw(self.new_dict({"tipo": self.new_string("VMError"), "mensagem": self.new_string(str(e))}))
            except Exception as e:
                self._throw(self.new_dict({"tipo": self.new_string(type(e).__name__), "mensagem": self.new_string(str(e))}))
            finally:
                # resolve pending after finally
                if self.handlers:
                    h = self.handlers[-1]
                    if h.state == 2 and h.frame_index == len(self.frames) - 1:
                        if frame.ip >= h.end_ip:
                            pending = h.pending
                            self.handlers.pop()
                            if pending is None:
                                continue
                            kind, val = pending
                            if kind == "throw":
                                self._throw(val)
                            elif kind == "return":
                                self.close_upvalues_from(frame.base)
                                base = frame.base
                                while len(self.stack) > base:
                                    self.pop()
                                self.frames.pop()
                                if not self.frames:
                                    return val
                                self.push(val)
                            else:
                                self._throw(self.new_dict({"tipo": self.new_string("VMError"), "mensagem": self.new_string("pending desconhecido")}))

# =====================================
# CLI
# =====================================

def main():
    args = sys.argv[1:]
    trace_gc = False
    if "--trace-gc" in args:
        trace_gc = True
        args.remove("--trace-gc")

    if len(args) < 1:
        print("Uso: python spyro-vm-2_2b.py [--trace-gc] arquivo.spy")
        sys.exit(2)

    entry_file = args[0]
    src = open(entry_file, "r", encoding="utf-8").read()

    toks = Lexer(src).tokenize()
    ast = Parser(toks).parse()

    comp = Compiler()
    entry_fn = comp.compile_program(ast)

    vm = VM(entry_file=entry_file, trace_gc=trace_gc)
    entry_clo = vm.new_closure(entry_fn, [])
    vm._run_closure(entry_clo, vm.root_globals, super_ref=None)

if __name__ == "__main__":
    main()