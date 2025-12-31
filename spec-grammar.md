# Especificação — Gramática (alto nível)

A gramática abaixo é intencionalmente “aproximada” (EBNF informal) para documentar a linguagem.

## Tokens (resumo)
- Identificadores: letras, `_`, dígitos (não iniciam com dígito)
- Literais: número, texto "...", lista [ ... ], mapa { ... }
- Palavras-chave: var, se, senao, enquanto, para, em, funcao, retorna,
  classe, tenta, captura, finalmente, lanca, fim, verdadeiro, falso, nulo

## Estruturas
programa      := { stmt }

stmt          := var_decl
              | assign
              | if_stmt
              | while_stmt
              | for_stmt
              | func_decl
              | class_decl
              | try_stmt
              | return_stmt
              | throw_stmt
              | expr_stmt

var_decl      := "var" IDENT "=" expr
assign        := (IDENT | index | prop) "=" expr

if_stmt       := "se" expr bloco [ "senao" bloco ] "fim"
while_stmt    := "enquanto" expr bloco "fim"
for_stmt      := "para" IDENT "em" expr bloco "fim"

func_decl     := "funcao" IDENT "(" [params] ")" bloco "fim"
params        := IDENT { "," IDENT }

class_decl    := "classe" IDENT [ ":" IDENT ] bloco "fim"

try_stmt      := "tenta" bloco "captura" IDENT bloco [ "finalmente" bloco ] "fim"
return_stmt   := "retorna" [expr]
throw_stmt    := "lanca" expr
expr_stmt     := expr

bloco         := { stmt }

expr          := or
or            := and { "ou" and }
and           := equality { "e" equality }

equality      := compare { ("==" | "!=") compare }
compare       := term { ("<" | "<=" | ">" | ">=") term }

term          := factor { ("+" | "-") factor }
factor        := unary { ("*" | "/" | "%") unary }
unary         := [ "nao" | "-" ] call

call          := primary { "(" [args] ")" | "[" expr "]" | "." IDENT }
args          := expr { "," expr }

primary       := NUMBER
              | STRING
              | "verdadeiro"
              | "falso"
              | "nulo"
              | IDENT
              | list
              | map
              | "(" expr ")"

list          := "[" [expr { "," expr }] "]"
map           := "{" [pair { "," pair }] "}"
pair          := (STRING | IDENT) ":" expr