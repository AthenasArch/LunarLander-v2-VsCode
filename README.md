### :wolf: Olá, sou o Leonardo Hilgemberg, criador da AthenasArch.
- Neste projeto vamos criar um agente para treinar o jogo LunarLander-V2, com o VsCode e Python no windows. Utilizaremos a API do jogo LunarLander da propria openAI, para não ter que reescrever todo o jogo.

---

Bibliotecas:
- gym[box2D]: Contém o ambiente LunarLander-v2 🌛 (utilizei a versão gym==0.21)
- stable-baselines3[extra]: A biblioteca de deep reinforcement learning.
- huggingface_sb3: Código adicional para Stable-baselines3 para carregar e fazer upload de modelos do repositori Hugging Face 🤗 Hub.

- A fonte dos estudos e práticas: https://simoninithomas.github.io/deep-rl-course/ 

---

# Descrição do funcionamento de uso do algoritmo
- Esse Script apresenta uma tela com Tk.
- Você tem 3 opções disponíveis:
   - 1 - Jogar Joga manualmente com as setas direcionais do teclado.
   - 2 - Teinar, aqui você inicia o treinamento do seu agente, recomendo excluir o meu arquivo .zip ja treeinado e iniciar o seu treinamento.
   - 3 - Testar, depois que possuir um agente treinado, você pode colocar para testar ele com a quantidade de episódios desejada.
   
   
   
   
# Como instalar e utilizar o LunarLander-v2 com o VsCode e python (testado no windows).
1. Adicionar o caminho da pasta do arquivo swigwin-4.1.1 à variável de ambiente PATH do Windows:
   - Recomendação: Mover a pasta para o diretório C:

2. Instalar os arquivos do Visual C++:
   - Acesse: https://visualstudio.microsoft.com/pt-br/visual-cpp-build-tools/

3. Instalar as bibliotecas necessárias para rodar o jogo:

4. Instalar box2d-py:
   - Execute: pip install box2d-py

5. Instalar a versão específica de pyglet:
   - Execute: pip3 install pyglet==1.5.21

6. Instalar a versão específica de gym:
   - Execute: pip install gym==0.21

7. Instalar PyOpenGL e PyOpenGL_accelerate:
   - Execute: pip install PyOpenGL PyOpenGL_accelerate

8. (Opcional) Criar um ambiente virtual e instalar todas as dependências nele:
   - Uma vez feito, feche o ambiente virtual. Isso pode ter influência no funcionamento do jogo.
