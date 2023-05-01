# adjuste from athenasarch
import sys
import pygame
import gym
from pygame.locals import *
import gym
from stable_baselines3.common.vec_env import DummyVecEnv
# from stable_baselines3.common.env_util import make_vec_env

# from huggingface_sb3 import package_to_hub

# Defina MANUAL_CONTROL como True para controle manual e False para IA
MANUAL_CONTROL = True

print(sys.version)

# Inicializa o ambiente Gym
env = gym.make("LunarLander-v2")
env.reset()

# Inicializa o pygame
pygame.init()
screen = pygame.display.set_mode((600, 400))
pygame.display.set_caption("Lunar Lander with Pygame")

clock = pygame.time.Clock()

while True:
    # Processa eventos do Pygame
    for event in pygame.event.get():
        if event.type == QUIT:
            # Fecha o ambiente Gym e o Pygame se o evento QUIT for recebido
            env.close()
            pygame.quit()
            sys.exit()

    # Define a ação a ser tomada com base no controle manual ou IA
    if MANUAL_CONTROL:
        # Obtém o estado das teclas pressionadas
        keys = pygame.key.get_pressed()

        # Define a ação com base nas teclas pressionadas
        if keys[K_LEFT]:
            print("Left Pres")
            action = 3
        elif keys[K_RIGHT]:
            print("right Pres")
            action = 1
        elif keys[K_UP]:
            print("up Pres")
            action = 2
        else:
            action = 0
    else:
        # Substitua esta linha pela ação gerada pela IA
        action = env.action_space.sample()

    # Atualiza o ambiente Gym com a ação escolhida
    _, _, done, _ = env.step(action)
    # Reinicia o ambiente se o episódio terminar
    if done:
        env.reset()

    # Renderiza o ambiente Gym e converte a imagem para um formato compatível com o Pygame
    frame = env.render(mode="rgb_array")
    frame = pygame.surfarray.make_surface(frame.swapaxes(0, 1))

    # Desenha o frame do Lunar Lander na janela do Pygame
    screen.blit(frame, (0, 0))
    pygame.display.flip()
    clock.tick(30)
