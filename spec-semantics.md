# Especificação — Semântica (alto nível)

## Escopos
- `var` declara no escopo atual.
- Funções criam novo escopo e podem capturar variáveis externas (closure).
- Blocos (`se`, `enquanto`, `para`, `tenta`) seguem a regra da VM (preferência: escopo léxico previsível).

## Avaliação
- Expressões avaliam da esquerda para a direita.
- Curto-circuito em `e` e `ou`.

## Mutabilidade
- Listas e mapas são mutáveis.
- Texto e números são imutáveis.

## Exceções
- `lanca valor` interrompe a execução até ser capturado.
- `tenta/captura/finalmente`:
  - executa bloco do `tenta`
  - se houver exceção, entra no `captura` (bind no nome)
  - `finalmente` sempre roda

## Funções
- `retorna` encerra a função (com valor opcional).
- Argumentos são passados por referência para objetos mutáveis (compartilham a mesma instância).

## Módulos
- `importa("x")` carrega `x` e retorna seu valor exportado.
- O carregamento deve ter cache (import do mesmo nome retorna o mesmo módulo).

## OO
- `classe` cria um objeto-classe chamável (construtor).
- Chamando a classe cria instância; se existir `__init__`, ele roda.
- `self` é passado implicitamente em chamadas de método.
- Herança: lookup de atributo procura na instância, depois na classe, depois na superclasse.