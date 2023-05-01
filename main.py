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

# Define a função 'train_model' para treinar o agente
def train_model(progress_text, render_training = False):

    # Define algumas variáveis de configuração
    env_id = "LunarLander-v2"  # Identificador do ambiente Gym
    my_model_name = "ppo-LunarLander-v2"
    model_architecture = "PPO"  # Nome do algoritmo de aprendizado por reforço
    repo_id = "seu_usuario/ppo-LunarLander-v2"  # Identificador do repositório Hugging Face Hub
    commit_message = "Upload PPO LunarLander-v2 trained agent"  # Mensagem de commit para o Hugging Face Hub
    token = "seu_token" # coloque seu toke aqui

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
    total_timesteps = 1000000 # quantidade de vezes que sera treinado 1M vezes treinado
    current_timesteps = 0
    update_interval = 10000 

    # Loop principal de treinamento
    while current_timesteps < total_timesteps:
        # Treina o modelo por 'update_interval' passos de tempo
        model.learn(total_timesteps=update_interval)

        # Atualiza a contagem de passos de tempo
        current_timesteps += update_interval

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

    # Salva o modelo treinado no diretório atual
    model.save(my_model_name)

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

# Função para iniciar o treinamento em uma nova thread
def start_training(progress_text, render_var):
    # Cria uma nova thread para executar a função 'train_model' e passa 'progress_text' como argumento
    train_thread = threading.Thread(target=train_model, args=(progress_text, render_var))
    # Inicia a thread
    train_thread.start()

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
    frame_y = 550
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

    # Cria um botão para iniciar o modo de controle manual
    manual_button = tk.Button(frame, text="Jogar manualmente", command=lambda: Process(target=manual_control).start(), width=20)
    manual_button.pack(pady=10)

    # Adiciona uma checkbox para renderizar o treinamento
    render_var = tk.BooleanVar() # variavel qeu recebe se renderiza ou nao
    render_check = tk.Checkbutton(frame, text="Renderizar treinamento", variable=render_var, fg="white", bg="black", selectcolor="black", activeforeground="white", activebackground="black")
    render_check.pack(pady=10)

    # Cria um botão para iniciar o treinamento do agente
    start_button = tk.Button(frame, text="Treinar agente", command=lambda: start_training(progress_var, render_var.get()), width=20)
    start_button.pack(pady=10)

    # Cria um botão para testar o agente treinado
    test_button = tk.Button(frame, text="Testar agente treinado", command=lambda: evaluate_trained_agent(), width=20)
    test_button.pack(pady=10)

    # Inicia o loop principal do tkinter
    root.mainloop()


# Inicia o programa chamando a função 'main'
if __name__ == "__main__":
    main()
