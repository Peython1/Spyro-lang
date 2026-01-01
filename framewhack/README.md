## Framewhack - Framework Bot para Tibia & Pokétibia

Framework em **Spyro** para criar bots de **Tibia** e **Pokétibia** com suporte a automação, handlers de eventos e adaptadores multi-servidor.

## Estrutura

framewhack/
├── core.spy                  # Engine do bot (eventos, estado, ciclo)
├── adapters/
│   └── tibia.spy            # Adaptador Tibia (conexão TCP, pacotes)
├── handlers/
│   └── tibia_handler.spy    # Lógica automática (auto-ataque, auto-cura)
├── examples/
│   └── tibia_bot_simples.spy # Bot básico com auto-combate
└── README.md                 # Esta documentação

## Como Usar

### 1. Bot Simples (Auto-Ataque)

```spy
var meu_bot = TibiaBot("MeuBot", "servidor.com", 7171)
var adapter = TibiaAdapter()

adapter.conectar(meu_bot.servidor, meu_bot.porta)

meu_bot.registrar_handler("criatura_avistada", funcao(evento)
    adapter.atacar(evento["id"])
fim)

## Auto-Cura 

meu_bot.registrar_handler("levou_dano", funcao(evento)
    se meu_bot.vida < 30 entao
        adapter.usar_magia("exura", meu_bot.nome)
    fim
fim)

Pokétibia (Mesma Base)
 Para Pokétibia, basta trocar a lógica:
Atacar → Capturar Pokémon
Magia → Items de captura
Combate → Batalha com atributos
A estrutura do framework reutiliza o mesmo protocolo TCP.


Eventos Disponíveis
conectado - Bot conectou ao servidor
desconectado - Bot desconectou
criatura_avistada - Detectou criatura/inimigo
levou_dano - Bot tomou dano
status_atualizado - Vida/Mana atualizadoserro - Erro de conexão/exec
uçãoRoadmap�Suporte a socket TC

Roadmap:
�Suporte a socket TCP nativo em Spyro
�Reverse engineering completo do protocolo Tibia
�Adaptador Pokétibia
�Sistema de rotas (pathfinding)
�Integração com banco de dados de criaturas
�Anti-detecção (comportamento humano)

Notas:
Framewhack é educacional; respeite os ToS dos servidores 🤣😏😈
Testado em Tibia 12.x e servidores custom (OTServer)
Pokétibia herda a mesma arquitetura
Status: Alpha

