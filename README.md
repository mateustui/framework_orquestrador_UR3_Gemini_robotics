# UR3 Robot Controller

Orquestrador baseado em LLM (Gemini Robotics-ER) para controle do robô UR3 no CoppeliaSim.

## Pré-requisitos

- Python 3.11+
- CoppeliaSim com ZMQ Remote API ativa

## Instalação

```bash
pip install -e .
```

## Configuração

Copie `.env.example` para `.env` e preencha:

```
GOOGLE_API_KEY="sua_chave_aqui" (criar em https://aistudio.google.com/api-keys)
SCENE_PATH="experimento-ur3.ttt"   # caminho para a cena do CoppeliaSim
```

## Uso

1. Abra o CoppeliaSim (não carregue a cena manualmente — o orquestrador faz isso).
2. Execute:

```bash
python run.py
```

3. Na janela que abrir, digite comandos em linguagem natural no campo **COMANDO** e pressione Enter ou clique **ENVIAR**.

## Exemplos de comandos

```
pegue o cubo vermelho e deposite no disco verde
detectar objetos
qual e o estado atual do robo?
```

Use o botão **RESET** para reiniciar a simulação sem fechar o programa.
