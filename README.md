# Spyro (linguagem) — Quickstart

Spyro é uma linguagem de script dinâmica em PT-BR para automação e bots, com sintaxe simples e blocos com “fim”.
Ela inclui: variáveis, controle de fluxo, funções/closures, módulos, exceções e OO (classes/instâncias/herança/super).

## Execução

Requer Python 3.10+.

Execute um script:
python spyro-vm-2_2b.py arquivo.spy

Opção útil:
python spyro-vm-2_2b.py --trace-gc arquivo.spy

## Exemplos (tudo junto)

Hello World:
escreva("Olá, mundo!")

Exceções:
tenta
    lanca {"tipo": "Erro", "mensagem": "falhou"}
captura e
    escreva("erro:", e.tipo, e.mensagem)
finalmente
    escreva("cleanup")
fim

Módulos:
var util = importa("util")
escreva(util.versao)

OO (classe/instância):
classe Pessoa
    funcao __init__(self, nome)
        self.nome = nome
    fim

    funcao fala(self)
        retorna "Oi, eu sou " + self.nome
    fim
fim

var p = Pessoa("Ana")
escreva(p.fala())

## Documentação
- Manual: reference.md
- Erros: errors.md
- Roadmap: roadmap.md
- Gramática: spec-grammar.md
- Semântica: spec-semantics.md
