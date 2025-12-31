# Referência (Spyro)

Este documento descreve a linguagem Spyro do ponto de vista do usuário: tipos, expressões, comandos, funções, módulos, exceções e OO.

> Convenção: palavras-chave em minúsculo, blocos terminam com `fim`.

## Tipos e valores
- nulo: `nulo`
- booleano: `verdadeiro`, `falso`
- número: inteiro/float (ex.: `1`, `2.5`)
- texto: `"abc"`
- lista: `[1, 2, 3]`
- mapa/objeto: `{"chave": "valor"}`
- função: valores chamáveis
- classes e instâncias (OO)

## Variáveis
var x = 10
x = x + 1

## Operadores (visão geral)
- Aritméticos: + - * / %
- Comparação: == != < <= > >=
- Lógicos: e, ou, nao
- Acesso: índice a[0] e propriedade obj.nome

## Controle de fluxo

Se / senao:
se x > 10
    escreva("maior")
senao
    escreva("nao maior")
fim

Enquanto:
var i = 0
enquanto i < 3
    escreva(i)
    i = i + 1
fim

Para:
para i em [1,2,3]
    escreva(i)
fim

## Funções
funcao soma(a, b)
    retorna a + b
fim

## Exceções
tenta
    lanca {"tipo":"Erro","mensagem":"x"}
captura e
    escreva(e.tipo, e.mensagem)
finalmente
    escreva("fim")
fim

## Módulos
var m = importa("nome_do_modulo")

## OO (classes)
- classe Nome ... fim
- construtor convencional: __init__(self, ...)
- métodos recebem self

classe A
    funcao ola(self)
        retorna "oi"
    fim
fim

## Builtins (mínimo)
- escreva(...): imprime
- importa(nome): carrega módulo