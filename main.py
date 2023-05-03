# From Leonardo by AthenasArch

import sys
import pygame
from pygame.locals import *
import gym
import tkinter as tk
from multiprocessing import Process
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
import threading
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from huggingface_sb3 import package_to_hub
from huggingface_hub import login
from stable_baselines3.common.evaluation import evaluate_policy
from tkinter import PhotoImage
from PIL import Image, ImageTk
import csv
import matplotlib.pyplot as plt
import os

learning_data = []

def print_final_position(observation):

    x_pos = observation[0]
    y_pos = observation[1]
    horizontal_speed = observation[2]
    vertical_speed = observation[3]
    angle = observation[4]
    angular_speed = observation[5]
    left_leg_contact = observation[6]
    right_leg_contact = observation[7]

    print("Coordenada pad horizontal (x) = ", x_pos)
    print("Coordenada pad vertical...(y) = ", y_pos)
    print("Velocidade horizontal.....(x) = ", horizontal_speed)
    print("Velocidade vertical.......(y) = ", vertical_speed)
    print("Ângulo....................... = ", angle)
    print("velocidade angular........... = ", angular_speed)
    print("perna esquerda tocou a terra. = ", left_leg_contact)
    print("perna direita tocou a terra...= ", right_leg_contact)

def save_learning_data(file_name):
    with open(file_name, "w", newline="") as csvfile:
        fieldnames = ["progress", "mean_reward"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for data in learning_data:
            writer.writerow(data)

# carrega modelo treinado anteriormente
def load_pretrained_model(model_name):
    try:
        loaded_model = PPO.load(model_name)
        return loaded_model
    except FileNotFoundError:
        return None

# compara o modelo treinado anteriormente com o novo para ver se houve melhoria
def compare_models(model, model_name):
    pretrained_model = load_pretrained_model(model_name)
    if pretrained_model:
        pretrained_reward = evaluate_current_model(pretrained_model)
        new_reward = evaluate_current_model(model)
        if new_reward > pretrained_reward:
            return True
        else:
            return False
    else:
        return True  # Se não houver modelo pré-treinado, consideramos o novo modelo como melhor


# Define a função 'train_model' para treinar o agente
def train_model(progress_text, evaluation_var, improvement_var, render_training = False):

    # Define algumas variáveis de configuração
    env_id = "LunarLander-v2"  # Identificador do ambiente Gym
    my_model_name = "ppo-LunarLander-v2" # o nome do modelo treinado a ser salvo
    model_architecture = "PPO"  # Nome do tipo algoritmo de aprendizado por reforço
    repo_user_name = os.environ.get("LUNARLANDER_REPO_ID")
    if repo_user_name is not None:
        repo_id = repo_user_name + "/ppo-LunarLander-v2"  # Identificador do repositório Hugging Face Hub
    else:
        repo_id = None
    commit_message = "Upload PPO LunarLander-v2 trained agent"  # Mensagem de commit para o Hugging Face Hub
    huggingface_token = os.environ.get("HUGGINGFACE_TOKEN")
    token = huggingface_token # coloque seu toke aqui
    print("\r\nYour Repo  ID: ", repo_user_name)
    print("\r\nYour Token ID: ", token)


    # HUGGINGFACE_TOKEN
    # LUNARLANDER_REPO_ID

    # Autenticar com o Hugging Face Hub
    login(token)

    # Cria um ambiente de avaliação para renderizar o agente durante o treinamento
    eval_env = gym.make(env_id)

    # Cria um ambiente vetorizado com 16 instâncias do ambiente LunarLander-v2
    env = make_vec_env(env_id, n_envs=16)

    # Cria um modelo PPO com a política MlpPolicy e o ambiente vetorizado criado acima
    # model = PPO("MlpPolicy", env, verbose=1) # estava rodando bem com isso
    # SOLUTION
    # We added some parameters to accelerate the training
    model = PPO(
        policy="MlpPolicy",
        env=env,
        n_steps=1024,
        batch_size=64,
        n_epochs=4,
        gamma=0.999,
        gae_lambda=0.98,
        ent_coef=0.01,
        verbose=1,
    )

    # Define o número total de passos de tempo para o treinamento e a frequência de atualização do progresso
    total_timesteps = 4000000 # quantidade de vezes que sera treinado 1M vezes treinado
    current_timesteps = 0 # somador do passo atual
    update_interval = 10000 # tempo para calculo percentual e rederizacao do treinamento

    # Loop principal de treinamento
    while current_timesteps < total_timesteps:
        # Treina o modelo por 'update_interval' passos de tempo
        model.learn(total_timesteps=update_interval)

        # Atualiza a contagem de passos de tempo
        current_timesteps += update_interval

        # Avalia o modelo treinado
        mean_reward, std_reward = evaluate_policy(model, DummyVecEnv([lambda: gym.make(env_id)]), n_eval_episodes=5, deterministic=True)

        # Armazena a recompensa média e o progresso atual em uma lista
        learning_data.append({"progress": current_timesteps / total_timesteps, "mean_reward": mean_reward})

        # Avalia o modelo atual e atualiza a variável de avaliação
        mean_reward = evaluate_current_model(model)
        evaluation_var.set(f"Recompensa média: {mean_reward:.2f}")

        # Atualiza a variável de progresso e mostra na interface do usuário
        progress = 100 * current_timesteps / total_timesteps
        progress_text.set(f"Progress: {progress:.2f}%")

        if(render_training):
            # Renderiza o ambiente de avaliação para visualizar o desempenho do agente
            obs = eval_env.reset()
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, done, _ = eval_env.step(action)
                eval_env.render()

    # Fecha o ambiente de avaliação
    eval_env.close()

    # Avalia o modelo atual e atualiza a variável de avaliação
    mean_reward = evaluate_current_model(model)
    evaluation_var.set(f"Recompensa média: {mean_reward:.2f}")

    if compare_models(model, my_model_name):
        improvement_var.set("Melhoria no modelo treinado!")

        # Salva o modelo treinado no diretório atual
        model.save(my_model_name)

        # Salva os dados de aprendizado em um arquivo CSV
        save_learning_data("learning_data.csv")

        # Envia o modelo treinado para o Hugging Face Hub
        package_to_hub(
            model=model,  # Our trained model
            model_name=my_model_name,  # The name of our trained model
            model_architecture=model_architecture,  # The model architecture we used: in our case PPO
            env_id=env_id,  # Name of the environment
            eval_env=DummyVecEnv([lambda: gym.make(env_id)]),  # Evaluation Environment
            repo_id=repo_id,  # id of the model repository from the Hugging Face Hub (repo_id = {organization}/{repo_name} for instance ThomasSimonini/ppo-LunarLander-v2
            commit_message=commit_message,
        )
    else:
        improvement_var.set("O novo treinamento não foi melhor que o anterior")



# Função para iniciar o treinamento em uma nova thread
def start_training(progress_text, evaluation_var, improvement_var, render_var):
    # Cria uma nova thread para executar a função 'train_model' e passa 'progress_text' como argumento
    train_thread = threading.Thread(target=train_model, args=(progress_text, evaluation_var, improvement_var, render_var))
    # Inicia a thread
    train_thread.start()

# aqui plotamos o grafico de aprendizado em funcao do percentual de tempo
def plot_learning_graph():
    progress = []
    mean_reward = []

    with open("learning_data.csv", "r") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            progress.append(float(row["progress"]))
            mean_reward.append(float(row["mean_reward"]))

    plt.plot(progress, mean_reward)
    plt.xlabel("Progresso de Treinamento")
    plt.ylabel("Recompensa Média")
    plt.title("Gráfico de Aprendizado")
    plt.show()

# avalia o treinamento do agente durante a execucao do script
def evaluate_current_model(model, n_eval_episodes=5):
    eval_env = gym.make("LunarLander-v2")
    mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=n_eval_episodes, deterministic=True)
    eval_env.close()
    return mean_reward


# aqui vamos testar o agente treinado
def evaluate_trained_agent():
    
    model_name = "ppo-LunarLander-v2" 
    n_eval_episodes = 10
    
    # Carrega o modelo treinado
    trained_model = PPO.load(model_name)

    # Cria um ambiente de avaliação
    eval_env = gym.make("LunarLander-v2")

    # Avalia o modelo treinado
    mean_reward, std_reward = evaluate_policy(trained_model, eval_env, n_eval_episodes=n_eval_episodes, deterministic=True)
    print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")

    # Renderiza o agente treinado
    for _ in range(n_eval_episodes):
        obs = eval_env.reset()
        done = False
        while not done:
            action, _ = trained_model.predict(obs, deterministic=True)
            obs, _, done, _ = eval_env.step(action)
            eval_env.render()

    # Fecha o ambiente de avaliação
    eval_env.close()

# Função para executar o controle manual do Lunar Lander
def manual_control():
    # Cria o ambiente Lunar Lander
    env = gym.make("LunarLander-v2")
    observation = env.reset()

    # Inicializa o Pygame
    pygame.init()

    # Cria uma janela para o Pygame
    screen = pygame.display.set_mode((600, 400))
    pygame.display.set_caption("Lunar Lander with Pygame")

    # Cria um objeto de clock para controlar a taxa de atualização do jogo
    clock = pygame.time.Clock()

    # Loop principal do jogo
    while True:
        # Processa eventos do Pygame
        for event in pygame.event.get():
            # Se o evento for do tipo QUIT, fecha o ambiente e o Pygame, e encerra o programa
            if event.type == QUIT:
                env.close()
                pygame.quit()
                sys.exit()

        # Verifica quais teclas estão pressionadas
        keys = pygame.key.get_pressed()

        # Define a ação com base nas teclas pressionadas
        if keys[K_LEFT]:
            action = 3
        elif keys[K_RIGHT]:
            action = 1
        elif keys[K_UP]:
            action = 2
        else:
            action = 0

        # Realiza a ação e verifica se o episódio terminou
        observation, reward, done, info = env.step(action)
        if done:
            observation = env.reset()
            print("Reompensa: ", reward, " - Reseta o ambiente.")
            print_final_position(observation)
            

        # Renderiza o ambiente e cria uma superfície do Pygame com base na renderização
        frame = env.render(mode="rgb_array")
        frame = pygame.surfarray.make_surface(frame.swapaxes(0, 1))

        # Desenha a superfície na tela e atualiza a janela do Pygame
        screen.blit(frame, (0, 0))
        pygame.display.flip() # precisa invertar a tela do jogo
        clock.tick(30)

# Função principal para executar o programa
def main():

    frame_x = 500
    frame_y = 700
    # Cria a janela principal do tkinter
    root = tk.Tk()
    root.title("Lunar Lander")
    root.configure(bg="black")
    root.minsize(width=frame_x, height=frame_y)
    root.maxsize(width=frame_x, height=frame_y)

    # Carrega a imagem e redimensiona
    image = Image.open("logo.png") # Insira o nome correto do arquivo de imagem
    image = image.resize((200, 200), Image.ANTIALIAS) # Ajuste os valores para redimensionar a imagem
    logo_image = ImageTk.PhotoImage(image)

    # Cria um frame para conter os elementos da interface
    frame = tk.Frame(root, bg="black")
    frame.pack(padx=10, pady=10)

    # Adiciona a imagem do logo
    logo_label = tk.Label(frame, image=logo_image, bg="black")
    logo_label.pack(pady=10)

    # Cria uma variável para exibir o progresso do treinamento
    progress_var = tk.StringVar()
    progress_label = tk.Label(frame, textvariable=progress_var)
    progress_label.pack(pady=10)

    # Cria uma variável para exibir a recompensa média
    evaluation_var = tk.StringVar()
    evaluation_label = tk.Label(frame, textvariable=evaluation_var)
    evaluation_label.pack(pady=10)

    # Cria uma variável para exibir a mensagem de melhoria do modelo
    improvement_var = tk.StringVar()
    improvement_label = tk.Label(frame, textvariable=improvement_var)
    improvement_label.pack(pady=10)

    # Cria um botão para iniciar o modo de controle manual
    manual_button = tk.Button(frame, text="Jogar manualmente", command=lambda: Process(target=manual_control).start(), width=20)
    manual_button.pack(pady=10)

    # Adiciona uma checkbox para renderizar o treinamento
    render_var = tk.BooleanVar() # variavel qeu recebe se renderiza ou nao
    render_check = tk.Checkbutton(frame, text="Renderizar treinamento", variable=render_var, fg="white", bg="black", selectcolor="black", activeforeground="white", activebackground="black")
    render_check.pack(pady=10)

    # Cria um botão para iniciar o treinamento do agente
    start_button = tk.Button(frame, text="Treinar agente", command=lambda: start_training(progress_var, evaluation_var, improvement_var, render_var.get()), width=20)
    start_button.pack(pady=10)

    # Cria um botão para testar o agente treinado
    test_button = tk.Button(frame, text="Testar agente treinado", command=lambda: evaluate_trained_agent(), width=20)
    test_button.pack(pady=10)

    # Cria um botão para plotar o gráfico de aprendizado
    plot_button = tk.Button(frame, text="Plotar gráfico de aprendizado", command=plot_learning_graph, width=20)
    plot_button.pack(pady=10)
    
    # Inicia o loop principal do tkinter
    root.mainloop()


# Inicia o programa chamando a função 'main'
if __name__ == "__main__":
    main()



# policy: 
#   O tipo de política a ser usada pelo agente. Neste caso, é uma política 
# baseada em Multilayer Perceptron (MLP), que é uma arquitetura de rede neural 
# feedforward.

# env: 
#   O ambiente do Gym no qual o agente será treinado. Neste caso, é a instância 
# do ambiente Lunar Lander.

# n_steps: 
#   O número de etapas de interação (ciclo ação-observação) com o ambiente
# que o agente coleta antes de atualizar os parâmetros da política. Um valor maior 
# pode melhorar a estabilidade do treinamento, mas também aumenta o tempo necessário 
# para cada atualização.

# batch_size: 
#   O tamanho do lote usado para atualizar os parâmetros da política. Um lote é um 
# subconjunto de dados de interação (ações, observações, recompensas) usados para 
# treinar a rede neural.

# n_epochs: 
#   O número de vezes que o algoritmo passa por todo o conjunto de dados 
# (etapas de interação) durante o treinamento. Cada passagem completa pelos dados 
# é chamada de época.

# gamma: 
#   O fator de desconto utilizado no cálculo da recompensa descontada acumulada. 
# Um valor mais próximo de 1 indica que o agente leva mais em consideração as recompensas 
# futuras, enquanto um valor mais baixo dá mais peso às recompensas imediatas.

# gae_lambda: 
#   O parâmetro lambda usado no cálculo da Generalized Advantage Estimation (GAE). 
# A GAE é uma técnica usada para estimar a vantagem de uma ação, que é a diferença 
# entre a recompensa acumulada esperada e a função valor do estado atual.

# ent_coef: 
#   Coeficiente de entropia usado para incentivar a exploração durante o treinamento. 
# Um valor mais alto incentiva o agente a explorar mais o ambiente, enquanto um valor 
# mais baixo o encoraja a explorar menos e se concentrar em ações conhecidas.

# verbose: 
#   Controla a quantidade de informações de depuração impressas durante o 
# treinamento. Neste caso, o valor 1 indica que apenas informações básicas serão impressas.