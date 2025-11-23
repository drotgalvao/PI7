# Sistema de Detecção e Recorte de Peças de Xadrez usando Visão Computacional

## 1. Contexto e Objetivo

Este projeto implementa um MVP (Produto Mínimo Viável) de um sistema de visão computacional para tabuleiros de xadrez.

O objetivo principal é:

- Detectar automaticamente as peças em uma imagem de tabuleiro de xadrez.
- Recortar cada peça em uma imagem individual.
- Salvar esses recortes de forma organizada, criando uma base de imagens para uso posterior (treinamento de modelos, análises, etc.).

Além disso, o projeto inclui ferramentas auxiliares para gerar tabuleiros sintéticos e tratar casos específicos de difícil detecção, como a rainha branca em casa clara.

---

## 2. Descrição Geral da Solução

A solução é composta por scripts em Python utilizando principalmente OpenCV e NumPy.

O fluxo geral é:

1. O usuário fornece uma imagem de um tabuleiro de xadrez.
2. O sistema executa a detecção de peças escuras (pretas) e claras (brancas) com base em limiares de intensidade em escala de cinza.
3. As detecções são filtradas por tamanho e refinadas por Non-Maximum Suppression (NMS) para remover caixas de detecção sobrepostas.
4. Cada contorno detectado é mapeado para uma fileira (rank) do tabuleiro.
5. Com base na posição inicial das peças de xadrez, o sistema associa cada detecção a um tipo de peça (torre, cavalo, bispo, dama, rei, peão) e salva os recortes em arquivos de imagem, como `black_queen.png`, `white_pawn.png` etc.
6. Para o caso específico da rainha branca, existe um script dedicado que utiliza a posição conhecida e processamento extra para extrair a peça com fundo transparente.

---

## 3. Principais Componentes do Projeto

### 3.1 `main.py`

Script principal de linha de comando. Suas responsabilidades incluem:

- Leitura da imagem de tabuleiro com `utils.image_loader.load_image`.
- Chamada de `utils.detection.detect_pieces` duas vezes:
  - `polarity="dark"` para peças pretas.
  - `polarity="light"` para peças brancas, ajustando o limiar de cinza e tentando múltiplos valores quando há poucas detecções.
- Aplicação de `utils.nms.non_max_suppression` para remover detecções redundantes.
- Organização das peças por fileira (rank) utilizando `utils.detection.get_rank_for_y`.
- Mapeamento das detecções para a configuração inicial de um tabuleiro de xadrez (tipos de peças por rank e coluna).
- Modo automático (`--auto`), que salva, sem interface interativa, uma peça de cada tipo por cor em um diretório de saída (por padrão, `crops/`).
- Modo interativo, que usa `utils.crop_saver.interactive_crop` para o usuário escolher visualmente quais recortes salvar.

### 3.2 `utils/detection.py`

Módulo responsável pela lógica de visão computacional básica:

- `detect_pieces`: função de alto nível que delega para:
  - `detect_dark_contours`: detecção de contornos de peças escuras (`THRESH_BINARY_INV`).
  - `detect_light_contours`: detecção de contornos de peças claras (`THRESH_BINARY`).
- Uso de:
  - Conversão para escala de cinza (`cv2.cvtColor`).
  - Binarização por limiar (`cv2.threshold`).
  - Operações morfológicas de abertura e fechamento para redução de ruído (`cv2.morphologyEx`).
  - Extração de contornos (`cv2.findContours`).
- Filtro de contornos por área relativa (proporção da área da imagem) para descartar ruídos e regiões muito grandes.
- `get_rank_for_y`: dado um `y` na imagem e a altura total, calcula a fileira (0 a 7) em que a peça se encontra.
- `filter_by_rank`: permite filtrar detecções pertencentes a uma determinada fileira.

### 3.3 `generate_chessboard.py`

Gera uma imagem sintética de tabuleiro de xadrez:

- Função `generate_chessboard_image` que:
  - Cria uma matriz de pixels com NumPy.
  - Desenha as casas alternando cores claras e escuras com OpenCV (`cv2.rectangle`).
  - Opcionalmente desenha rótulos de arquivos (a–h) e ranks (1–8) nas bordas.
- Essa ferramenta é útil para testes de visualização e para produzir imagens de referência controladas.

### 3.4 `extract_white_queen.py`

Script específico para a extração da rainha branca:

- Utiliza coordenadas conhecidas da posição inicial da rainha branca no tabuleiro (coluna D, última fileira).
- Recorta inicialmente apenas a casa correspondente.
- Aplica:
  - Equalização de histograma (CLAHE) para melhorar contraste.
  - Detecção de bordas com Canny (`cv2.Canny`).
  - Operações morfológicas para unir bordas.
  - Detecção de contornos e seleção do maior (assumido como a peça).
- Cria um recorte com fundo transparente usando `utils.crop_saver.create_transparent_crop` e técnicas de flood fill para isolar o fundo.
- Salva o resultado como `crops/white_queen.png`, fornecendo uma imagem de alta qualidade da rainha branca com canal alfa.

### 3.5 Outros utilitários (`utils/image_loader.py`, `utils/crop_saver.py`, `utils/nms.py`)

- `image_loader.py`: abstrai a leitura/carregamento de imagens, centralizando tratamento de erros e formatos.
- `crop_saver.py`: oferece funções para:
  - Salvar recortes com padding.
  - Criar recortes com fundo transparente a partir de uma máscara.
  - Interagir com o usuário (modo interativo) para seleção manual de regiões.
- `nms.py`: implementa Non-Maximum Suppression (NMS), técnica padrão em visão computacional para remover múltiplas detecções muito sobrepostas, mantendo apenas as caixas mais representativas.

---

## 4. Tecnologias Utilizadas

- **Linguagem:** Python.
- **Bibliotecas principais:**
  - OpenCV (`cv2`) para processamento de imagens, limiarização, morfologia, contornos, desenho e manipulação de canais.
  - NumPy (`numpy`) para operações matriciais e criação de imagens base.
- **Execução:** scripts de linha de comando, recebendo parâmetros como caminho da imagem, diretório de saída e parâmetros de detecção (limiar de cinza, área mínima/máxima de contornos etc.).

---

## 5. Fluxo de Uso (MVP)

1. O usuário captura ou fornece uma foto de um tabuleiro de xadrez visto de cima.
2. Executa o script principal, por exemplo:

   ```pwsh
   python main.py caminho/para/tabuleiro.png --auto -o crops
   ```

3. O sistema:
   - Processa a imagem.
   - Detecta peças pretas e brancas.
   - Aplica filtros e NMS.
   - Mapeia as peças às posições iniciais.
   - Gera recortes nomeados no diretório `crops/`.
4. Opcionalmente, o usuário pode:
   - Usar o modo interativo para refinar recortes.
   - Rodar `extract_white_queen.py` para obter uma imagem específica da rainha branca com fundo transparente.
   - Gerar um tabuleiro sintético com `generate_chessboard.py` para testes.

---

## 6. Resultados Esperados do MVP

- Conjunto de imagens individuais das peças de xadrez (pretas e brancas), salvas com nomes padronizados no diretório `crops/`.
- Capacidade de lidar com diferentes níveis de iluminação por meio do ajuste de limiares e área mínima/máxima de contornos.
- Extração de um caso crítico (rainha branca em casa clara) usando um pipeline dedicado.
- Base funcional suficiente para:
  - Criar um dataset de peças.
  - Servir como etapa inicial de um sistema maior, por exemplo, reconhecimento automático de posição (FEN) em um tabuleiro real.

---

## 7. Limitações e Próximos Passos

- **Dependência do enquadramento:** o MVP assume que o tabuleiro está bem enquadrado e visto de cima, com as casas aproximadamente alinhadas.
- **Configuração inicial:** o mapeamento de peças por rank está alinhado com a posição inicial padrão do xadrez; outras posições/partidas exigiriam técnicas adicionais (classificação de peças, reconhecimento de símbolos etc.).
- **Sensibilidade à iluminação:** embora existam ajustes de limiar, variações extremas de luz e sombras podem prejudicar a detecção.

**Próximos passos possíveis:**

- Treinar um classificador de peças usando os recortes gerados.
- Reconhecer automaticamente a posição FEN a partir de uma foto do tabuleiro.
- Implementar correção de perspectiva para imagens capturadas em ângulos não ideais.
- Criar uma interface gráfica simples para carregar imagem, visualizar detecções e exportar resultados.
