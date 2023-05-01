import gym
import tkinter as tk
import threading
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from huggingface_sb3 import package_to_hub
import huggingface_hub


# Define a função 'train_model' para treinar o agente
def train_model(progress_var):
    # Define algumas variáveis de configuração
    env_id = "LunarLander-v2"  # Identificador do ambiente Gym
    model_architecture = "PPO"  # Nome do algoritmo de aprendizado por reforço
    repo_id = "athenasarch/ppo-LunarLander-v2"  # Identificador do repositório Hugging Face Hub
    commit_message = "Upload PPO LunarLander-v2 trained agent"  # Mensagem de commit para o Hugging Face Hub

    # Cria um ambiente vetorizado com 16 instâncias do ambiente LunarLander-v2
    env = make_vec_env(env_id, n_envs=16)

    # Cria um modelo PPO com a política MlpPolicy e o ambiente vetorizado criado acima
    model = PPO("MlpPolicy", env, verbose=1)

    # Define o número total de passos de tempo para o treinamento e a frequência de atualização do progresso
    total_timesteps = 1000000
    current_timesteps = 0
    update_interval = 10000

    # Cria um ambiente de avaliação para renderizar o agente durante o treinamento
    eval_env = gym.make(env_id)

    # Loop principal de treinamento
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

def main():
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

if __name__ == "__main__":
    main()




#sobre acelerar o treinamento:

# A linha env = make_vec_env(env_id, n_envs=16) cria um ambiente vetorizado com 16
#    instâncias paralelas do ambiente LunarLander-v2. Um ambiente vetorizado é uma 
#   maneira de executar várias instâncias de um ambiente em paralelo, o que pode 
#   acelerar o treinamento do agente.

# A função make_vec_env é fornecida pela biblioteca stable_baselines3 e aceita 
#   dois argumentos principais:

# env_id: Identificador do ambiente Gym que você deseja criar. Neste caso, 
#   é o "LunarLander-v2".
# n_envs: Número de instâncias paralelas do ambiente que você deseja criar. 
#   Neste caso, são 16 instâncias.
# Criar um ambiente vetorizado com 16 instâncias paralelas significa que 16 episódios
#  diferentes do ambiente LunarLander-v2 serão executados ao mesmo tempo. Isso permite
#  que o agente explore mais rapidamente o espaço de estados e ações do ambiente e, em
#  muitos casos, pode acelerar o processo de aprendizado.

# Ao treinar o agente, o modelo PPO interage com essas 16 instâncias paralelas, coletando 
# experiências e atualizando sua política de maneira mais eficiente.





# O progresso que aparece na tela é uma medida do progresso do treinamento, expresso como uma
#   porcentagem. Ele vai de 0% a 100%. O progresso é calculado com base no número atual de passos
#   de tempo (current_timesteps) em relação ao número total de passos de tempo (total_timesteps) 
#   definidos para o treinamento. O valor de total_timesteps é definido como 1.000.000 no código,
#   e o progresso é atualizado sempre que o agente aprende por mais update_interval passos de
#   tempo (10.000, neste caso).

# O ambiente do jogo é renderizado após cada atualização do progresso. Portanto, a cada 10.000 passos
#   de tempo (aproximadamente), o jogo é renderizado na tela. No entanto, você pode ajustar a 
#   frequência de renderização modificando o valor de update_interval no código. Note que a renderização 
#   do jogo pode tornar o treinamento mais lento, então você deve encontrar um equilíbrio entre a 
#   frequência de renderização e a velocidade de treinamento.



# A explicacao generica sobre o algoritmo:

# O algoritmo utilizado neste código é o PPO (Proximal Policy Optimization), que é um algoritmo de 
#   aprendizado por reforço desenvolvido pela OpenAI. O objetivo do PPO é encontrar uma política ótima 
#   para um agente, que é uma função que determina a ação a ser tomada em cada estado do ambiente. O PPO 
#   faz isso através da otimização da política em relação a uma função de custo, que mede o desempenho do 
#   agente no ambiente. O PPO é um algoritmo de aprendizado por reforço de política, o que significa que 
#   ele otimiza diretamente a política, em vez de aprender uma função de valor e derivar a política a 
#   partir dela.

# Para aplicar esse algoritmo a outros jogos, você precisará seguir os passos abaixo:

# Encontre o nome do ambiente do jogo que você deseja treinar no Gym (um framework para desenvolver e 
#   comparar algoritmos de aprendizado por reforço). A lista de ambientes disponíveis pode ser 
#   encontrada aqui: https://gym.openai.com/envs/
# Substitua a variável env_id no código com o nome do ambiente escolhido. Por exemplo, se você deseja 
#   treinar o agente no jogo Breakout, use env_id = "Breakout-v0".
# Ajuste os hiperparâmetros do algoritmo, como o número total de passos de tempo e o intervalo de 
#   atualização, conforme necessário.
# Verifique se a política utilizada é adequada para o ambiente escolhido. Neste exemplo, estamos usando 
#   a política MlpPolicy. Dependendo do ambiente, você pode precisar usar uma política diferente, como 
#   CnnPolicy para ambientes baseados em imagens.
# O algoritmo PPO pode ser aplicado a uma ampla variedade de jogos e problemas, desde que o ambiente 
#   possa ser modelado como um Processo de Decisão de Markov (MDP). Os jogos que podem ser resolvidos 
#   com PPO incluem, mas não estão limitados a, jogos de Atari, robótica e controle contínuo, como
#   LunarLander, CartPole, Pendulum e muitos outros.

# É importante notar que os hiperparâmetros e a política podem precisar ser ajustados para obter um bom 
#   desempenho em diferentes jogos. Além disso, o treinamento pode levar mais tempo em ambientes mais 
#   complexos.