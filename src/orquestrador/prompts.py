_FUNCOES_DISPONIVEIS = """
def move_to_position(x: float, y: float, z: float, high: bool = True):
    '''Move o end-effector para coordenadas XYZ.
    high=True mantem altura segura durante movimento.'''

def move_to_pose(x: float, y: float, z: float, rx: float, ry: float, rz: float):
    '''Move para pose completa (posicao + orientacao em graus).'''

def move_joint(joint: int, delta: float):
    '''Gira junta especifica (0-5) por delta graus.
    Juntas: 0=base, 1=ombro, 2=cotovelo, 3=punho1, 4=punho2, 5=punho3'''

def open_gripper():
    '''Abre a garra para soltar objetos.'''

def close_gripper():
    '''Fecha a garra para segurar objetos.'''

def go_home():
    '''Retorna a posicao inicial segura.'''

def detect_objects():
    '''Usa visao estereo para detectar e localizar objetos na cena.
    Retorna lista de objetos com posicoes 3D.'''

def locate_object(name: str):
    '''Localiza objeto especifico pelo nome.
    Retorna posicao 3D se encontrado.'''

def pick_object(object_name: str):
    '''Sequencia completa para pegar objeto:
    1. Localiza objeto
    2. Move acima do objeto
    3. Desce ate objeto
    4. Fecha garra
    5. Sobe com objeto'''

def place_at_position(x: float, y: float, z: float):
    '''Coloca objeto segurado em posicao especifica.'''

def place_on_object(target_name: str):
    '''Coloca objeto segurado sobre outro objeto.
    Localiza o destino automaticamente.'''

def get_robot_state():
    '''Retorna estado atual: posicao, orientacao e configuracao das juntas.'''

def wait(seconds: float):
    '''Aguarda tempo especificado antes de proxima acao.'''

def save_object_position(name: str, key: str):
    '''Localiza um objeto pelo nome e salva sua posicao 3D na memoria.
    key e um apelido para reutilizar depois (ex: "pos_verde").'''

def place_at_saved(key: str):
    '''Coloca o objeto segurado na posicao salva anteriormente com save_object_position.'''
"""

_FORMATO_RESPOSTA = """
Analise o comando e retorne um JSON com:
{{
    "entendido": true/false,
    "explicacao": "breve explicacao do que sera feito",
    "acoes": [
        {{"funcao": "nome_funcao", "args": {{"arg1": valor, ...}}}},
        ...
    ]
}}

Use sempre os mesmos label de retorno para montar a sequencia de acoes.

Se o comando for uma pergunta ou nao requerer acao, retorne:
{{
    "entendido": true,
    "explicacao": "resposta a pergunta",
    "acoes": []
}}
"""

ROBOT_API_SCHEMA = (
    """
Voce controla um robo UR3 com garra atraves de uma API.
Analise o comando do usuario e retorne uma sequencia de chamadas de funcao. Sempre retone usando os labels em portugues brasileiro que foi definido e nunca utilize acentuacao.

IMPORTANTE: "posicao original" nao e nome de objeto. Para reutilizar posicoes, use save_object_position e place_at_saved.


FUNCOES DISPONIVEIS:
"""
    + _FUNCOES_DISPONIVEIS
    + """
COMANDO DO USUARIO: "{command}"
"""
    + _FORMATO_RESPOSTA
)

ROBOT_API_SCHEMA_WITH_VISION = (
    """
Voce controla um robo UR3 com garra atraves de uma API.
Analise o comando do usuario e retorne uma sequencia de chamadas de funcao. Sempre retorne usando os labels em portugues brasileiro que foram definidos e nunca utilize acentuacao.

REGRAS:
- NAO inclua go_home na lista de acoes. O retorno para HOME e executado automaticamente.
- "posicao original" nao e nome de objeto. Use save_object_position e place_at_saved para reutilizar posicoes.
- Use as posicoes da cena abaixo diretamente. Nao e necessario chamar detect_objects no inicio.

ESTADO ATUAL DA CENA (visao estereo):
{scene_description}

FUNCOES DISPONIVEIS:
"""
    + _FUNCOES_DISPONIVEIS
    + """
COMANDO DO USUARIO: "{command}"
"""
    + _FORMATO_RESPOSTA
)


PROMPT_DETECTAR_OBJETOS = """
Analise esta imagem de uma cena robotica industrial.
Identifique TODOS os objetos visiveis, exceto o robo.

Para cada objeto, forneca:
- Um label descritivo em PORTUGUES BRASILEIRO SEM acentos ou caracteres especiais (ex: "cubo vermelho", "esfera azul", "circulo verde", "cilindro amarelo")
- As coordenadas do centro do objeto

Retorne APENAS JSON valido no formato:
[{"point": [y, x], "label": "descricao do objeto em portugues sem acentos"}, ...]

Coordenadas normalizadas de 0 a 1000.
Se nao houver objetos, retorne: []
"""

PROMPT_LOCALIZAR_CAMERA2 = """
Esta e a mesma cena vista de outro angulo.
Localize EXATAMENTE estes objetos: {labels}

Para cada objeto da lista, encontre o centro na imagem.
Use os MESMOS labels fornecidos.

Retorne APENAS JSON:
[{{"point": [y, x], "label": "nome_exato"}}, ...]

Coordenadas normalizadas de 0 a 1000.
"""
