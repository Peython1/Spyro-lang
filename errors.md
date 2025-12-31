# Erros (Spyro)

Este arquivo lista erros comuns, causa provável e como corrigir.

## Erro de sintaxe

Sintoma:
- Mensagem indicando token inesperado ou bloco não fechado.

Causas comuns:
- Esqueceu `fim` em `se`, `enquanto`, `para`, `funcao`, `classe`, `tenta`.
- Usou palavra-chave errada (ex.: `senão` com acento em vez de `senao`).

Como resolver:
- Confira o balanceamento de blocos e a indentação.
- Simplifique o trecho até compilar.

## Erro de nome (variável/função não definida)

Sintoma:
- Mensagem tipo “nome não definido”.

Causas comuns:
- Variável sem `var` no primeiro uso.
- Erro de digitação.
- Tentou usar variável fora do escopo.

Como resolver:
- Declare com `var`.
- Verifique o escopo: variável declarada dentro de função/bloco não existe fora.

## Erro de tipo

Sintoma:
- Operação inválida (ex.: somar texto com número sem conversão).

Como resolver:
- Converta para texto quando necessário.
- Garanta o tipo esperado antes da operação.

## Exceção lançada pelo usuário

Sintoma:
- Um valor lançado por `lanca` interrompe o fluxo.

Como resolver:
- Envolva com `tenta/captura`.
- Padronize o objeto de erro: `{"tipo": "...", "mensagem": "..."}`.