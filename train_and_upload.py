# import gym
# from stable_baselines3 import PPO
# from stable_baselines3.common.vec_env import DummyVecEnv
# from stable_baselines3.common.env_util import make_vec_env
# from huggingface_sb3 import package_to_hub


# env_id = "LunarLander-v2"
# model_architecture = "PPO"
# repo_id = "athenasarch/ppo-LunarLander-v2"  # Substitua 'your_huggingface_username' pelo seu nome de usuário do Hugging Face Hub.
# commit_message = "Upload PPO LunarLander-v2 trained agent"

# # Crie o ambiente
# env = make_vec_env(env_id, n_envs=4)

# # Instancie o agente PPO
# model = PPO("MlpPolicy", env, verbose=1)

# # Treine o agente
# model.learn(total_timesteps=1000000)

# # Crie o ambiente de avaliação
# eval_env = DummyVecEnv([lambda: gym.make(env_id)])

# # Faça o upload do modelo treinado para o Hugging Face Hub
# package_to_hub(
#     model=model,
#     model_name=model_architecture,
#     model_architecture=model_architecture,
#     env_id=env_id,
#     eval_env=eval_env,
#     repo_id=repo_id,
#     commit_message=commit_message,
# )


# import gym
# import tkinter as tk
# import sys
# import threading
# from stable_baselines3 import PPO
# from stable_baselines3.common.vec_env import DummyVecEnv
# from stable_baselines3.common.env_util import make_vec_env
# from huggingface_sb3 import package_to_hub


# class TextRedirector:
#     def __init__(self, widget):
#         self.widget = widget

#     def write(self, text):
#         self.widget.delete(1.0, tk.END)
#         self.widget.insert(tk.END, text)
#         self.widget.see(tk.END)

#     def flush(self):
#         pass


# def train_model(progress_var, speed_var):
#     env_id = "LunarLander-v2"
#     model_architecture = "PPO"
#     repo_id = "athenasarch/ppo-LunarLander-v2"
#     commit_message = "Upload PPO LunarLander-v2 trained agent"

#     env = make_vec_env(env_id, n_envs=35)
#     model = PPO("MlpPolicy", env, verbose=1)

#     total_timesteps = 1000000
#     current_timesteps = 0
#     update_interval = 10000

#     while current_timesteps < total_timesteps:
#         model.learn(total_timesteps=update_interval)
#         current_timesteps += update_interval

#         progress = 100 * current_timesteps / total_timesteps
#         progress_var.set(f"Progress: {progress:.2f}%")

#         fps = model.get_vec_normalize_env().get_attr("fps")[0]
#         speed_var.set(f"Speed: {fps} fps")

#     eval_env = DummyVecEnv([lambda: gym.make(env_id)])

#     package_to_hub(
#         model=model,
#         model_name=model_architecture,
#         model_architecture=model_architecture,
#         env_id=env_id,
#         eval_env=eval_env,
#         repo_id=repo_id,
#         commit_message=commit_message,
#     )


# def start_training(progress_var, speed_var):
#     train_thread = threading.Thread(target=train_model, args=(progress_var, speed_var))
#     train_thread.start()


# root = tk.Tk()
# root.title("Training Progress")

# frame = tk.Frame(root)
# frame.pack(padx=10, pady=10)

# text_area = tk.Text(frame, wrap=tk.WORD, width=40, height=25)
# text_area.pack(pady=10)

# sys.stdout = TextRedirector(text_area)

# progress_var = tk.StringVar()
# progress_label = tk.Label(frame, textvariable=progress_var)
# progress_label.pack(pady=10)

# speed_var = tk.StringVar()
# speed_label = tk.Label(frame, textvariable=speed_var)
# speed_label.pack(pady=10)

# start_button = tk.Button(frame, text="Start Training", command=lambda: start_training(progress_var, speed_var))
# start_button.pack(pady=10)

# root.mainloop()


import gym
import tkinter as tk
import threading
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from huggingface_sb3 import package_to_hub
import huggingface_hub


# treina o modelo
def train_model(progress_var):
    # Define algumas variáveis de configuração
    env_id = "LunarLander-v2"
    model_architecture = "PPO"
    repo_id = "athenasarch/ppo-LunarLander-v2"
    commit_message = "Upload PPO LunarLander-v2 trained agent"

    # Cria um ambiente de treinamento vetorizado com 16 instâncias do ambiente LunarLander-v2
    env = make_vec_env(env_id, n_envs=16)

    # Cria um modelo PPO com a política MlpPolicy e o ambiente vetorizado criado acima
    model = PPO("MlpPolicy", env, verbose=1)

    # Define o número total de passos de tempo para o treinamento e a frequência de atualização do progresso
    total_timesteps = 1000000
    current_timesteps = 0
    update_interval = 10000

    # Cria um ambiente de avaliação para renderizar o agente durante o treinamento
    eval_env = gym.make(env_id)

    # Enquanto o agente não atingir o número total de passos de tempo
    while current_timesteps < total_timesteps:
        # Treina o modelo por 'update_interval' passos de tempo
        model.learn(total_timesteps=update_interval)

        # Atualiza a contagem de passos de tempo
        current_timesteps += update_interval

        # Atualiza a variável de progresso e mostra na interface do usuário
        progress = 100 * current_timesteps / total_timesteps
        progress_var.set(f"Progress: {progress:.2f}%")

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
    model.save("trained_model")

    # Envia o modelo treinado para o Hugging Face Hub
    package_to_hub(
        model=model,
        model_name=model_architecture,
        model_architecture=model_architecture,
        env_id=env_id,
        eval_env=DummyVecEnv([lambda: gym.make(env_id)]),
        repo_id=repo_id,
        commit_message=commit_message,
    )

# Função para iniciar o treinamento em uma nova thread
def start_training(progress_var):
    train_thread = threading.Thread(target=train_model, args=(progress_var,))
    train_thread.start()

# Criação da interface do usuário com Tkinter
root = tk.Tk()
root.title("Training Progress")
root.minsize(width=300, height=150)
root.maxsize(width=300, height=150)

frame = tk.Frame(root)
frame.pack(padx=10, pady=10)

progress_var = tk.StringVar()
progress_label = tk.Label(frame, textvariable=progress_var)
progress_label.pack(pady=10)

start_button = tk.Button(frame, text="Start Training", command=lambda: start_training(progress_var))
start_button.pack(pady=10)

root.mainloop()


# O progresso que aparece na tela é uma medida do progresso do treinamento, expresso como uma porcentagem. Ele vai de 0% a 100%. O progresso é calculado com base no número atual de passos de tempo (current_timesteps) em relação ao número total de passos de tempo (total_timesteps) definidos para o treinamento. O valor de total_timesteps é definido como 1.000.000 no código, e o progresso é atualizado sempre que o agente aprende por mais update_interval passos de tempo (10.000, neste caso).

# O ambiente do jogo é renderizado após cada atualização do progresso. Portanto, a cada 10.000 passos de tempo (aproximadamente), o jogo é renderizado na tela. No entanto, você pode ajustar a frequência de renderização modificando o valor de update_interval no código. Note que a renderização do jogo pode tornar o treinamento mais lento, então você deve encontrar um equilíbrio entre a frequência de renderização e a velocidade de treinamento.