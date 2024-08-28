import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from collections import deque
import os
import tkinter as tk
from tkinter import messagebox
import time

# Глобальные параметры
GAMMA = 0.95
LEARNING_RATE = 0.001
MEMORY_SIZE = 5000
BATCH_SIZE = 64
EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.99
TARGET_UPDATE_INTERVAL = 10

class DQN:
    def __init__(self, model_name):
        self.exploration_rate = EXPLORATION_MAX
        self.action_space = [i for i in range(9)]
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.model_name = model_name

        # Модель нейросети
        self.model = Sequential()
        self.model.add(Dense(64, input_shape=(9,), activation="relu"))
        self.model.add(Dense(64, activation="relu"))
        self.model.add(Dense(9, activation="linear"))
        self.model.compile(optimizer=tf.optimizers.Adam(learning_rate=LEARNING_RATE), loss='mse')

        # Загрузка модели, если она существует
        if os.path.exists(f'{self.model_name}.weights.h5'):
            self.model.load_weights(f'{self.model_name}.weights.h5')

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, available_actions):
        if np.random.rand() < self.exploration_rate:
            return random.choice(available_actions)
        q_values = self.model.predict(np.array([state]))
        valid_q_values = np.array([q_values[0][i] if i in available_actions else -float('inf') for i in range(9)])
        return np.argmax(valid_q_values)

    def experience_replay(self):
        if len(self.memory) < BATCH_SIZE:
            return

        batch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, next_state, done in batch:
            q_update = reward
            if not done:
                q_update = reward + GAMMA * np.amax(self.model.predict(np.array([next_state]))[0])

            q_values = self.model.predict(np.array([state]))
            q_values[0][action] = q_update

            self.model.fit(np.array([state]), q_values, verbose=0)

        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)

    def save_model(self):
        self.model.save_weights(f'{self.model_name}.weights.h5')

class TicTacToeGame:
    def __init__(self, root):
        self.root = root
        self.player_x = DQN("player_x")
        self.player_o = DQN("player_o")
        self.board = [''] * 9
        self.buttons = []
        self.human_role = 'X'  # Изначально человек играет за X
        self.first_player = 'human'  # Человек начинает первым
        self.human_turn = None  # Кто ходит первым
        self.game_over = False  # Отслеживание окончания игры
        self.create_widgets()

    def create_widgets(self):
        # Задаем темный индиго для фона окна
        self.root.config(bg='#1C1C2E')  # Темный индиго для фона окна

        for i in range(9):
            button = tk.Button(self.root, text='', font='normal 20 bold', height=3, width=6,
                               bg='#1C1C2E',  # Темный индиго для фона кнопок
                               fg='white',    # Цвет текста по умолчанию
                               bd=1,          # Толщина границ кнопок
                               highlightbackground='lightgray',  # Светлые разделения
                               command=lambda i=i: self.human_click(i))
            button.grid(row=i//3, column=i%3, padx=5, pady=5)  # Отступы между кнопками
            self.buttons.append(button)
        
        self.switch_button = tk.Button(self.root, text="Switch to AI vs AI", 
                                       bg='#1C1C2E',  # Темный индиго для кнопки переключения
                                       fg='white',    # Цвет текста
                                       bd=1,          # Толщина границы кнопки
                                       highlightbackground='lightgray',  # Светлые разделения
                                       command=self.switch_mode)
        self.switch_button.grid(row=3, columnspan=3, pady=10)

        self.reset_board()

    def reset_board(self):
        self.board = [''] * 9
        self.game_over = False
        for button in self.buttons:
            button.config(text='', fg='black')

        # Чередование того, кто ходит первым
        if self.first_player == 'human':
            self.human_turn = self.human_role
        else:
            self.human_turn = None
            self.root.after(500, self.ai_move)

    def update_board(self):
        for i in range(9):
            if self.board[i] == 'X':
                self.buttons[i].config(text='X', fg='#32CD32')  # Зеленый лайм для крестиков
            elif self.board[i] == 'O':
                self.buttons[i].config(text='O', fg='#FF2400')  # Алый цвет для ноликов

    def switch_mode(self):
        if self.switch_button["text"] == "Switch to AI vs AI":
            self.switch_button.config(text="Switch to Human vs AI")
            self.start_ai_vs_ai()
        else:
            self.switch_button.config(text="Switch to AI vs AI")
            self.reset_board()

    def human_click(self, index):
        if not self.game_over and self.human_turn == self.human_role and self.board[index] == '':
            self.board[index] = self.human_role
            self.update_board()
            winner, done = self.check_win(self.human_role)
            if done:
                self.end_game(winner)
            else:
                self.human_turn = None
                self.root.after(500, self.ai_move)

    def ai_move(self):
        if not self.game_over and self.human_turn is None:
            state = self.get_board_state()
            available_actions = [i for i, cell in enumerate(self.board) if cell == '']
            if self.human_role == 'X':
                action = self.player_o.act(state, available_actions)
                self.board[action] = 'O'
                self.update_board()
                winner, done = self.check_win('O')
                if done:
                    self.end_game(winner)
                else:
                    self.human_turn = 'X'
            else:
                action = self.player_x.act(state, available_actions)
                self.board[action] = 'X'
                self.update_board()
                winner, done = self.check_win('X')
                if done:
                    self.end_game(winner)
                else:
                    self.human_turn = 'O'

    def start_ai_vs_ai(self):
        self.reset_board()
        current_player = 'X'
        done = False

        while not done:
            self.root.update()
            state = self.get_board_state()
            available_actions = [i for i, cell in enumerate(self.board) if cell == '']

            if current_player == 'X':
                action = self.player_x.act(state, available_actions)
                self.board[action] = 'X'
                current_player = 'O'
            else:
                action = self.player_o.act(state, available_actions)
                self.board[action] = 'O'
                current_player = 'X'

            self.update_board()
            time.sleep(0.5)

            winner, done = self.check_win(current_player)
            if done:
                self.end_game(winner)

    def get_board_state(self):
        state = np.zeros(9)
        for i in range(9):
            if self.board[i] == 'X':
                state[i] = -1
            elif self.board[i] == 'O':
                state[i] = 1
        return state

    def check_win(self, player):
        win_combinations = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],
            [0, 3, 6], [1, 4, 7], [2, 5, 8],
            [0, 4, 8], [2, 4, 6]
        ]
        for combo in win_combinations:
            if self.board[combo[0]] == self.board[combo[1]] == self.board[combo[2]] == player:
                return player, True
        if '' not in self.board:
            return None, True  # Ничья
        return None, False

    def end_game(self, winner):
        self.game_over = True
        if winner:
            messagebox.showinfo("Игра окончена", f"Победил {winner}!")
        else:
            messagebox.showinfo("Игра окончена", "Ничья!")
        
        # Чередование ролей и первого хода
        self.human_role = 'O' if self.human_role == 'X' else 'X'
        self.first_player = 'ai' if self.first_player == 'human' else 'human'
        self.reset_board()

# Запуск игры
if __name__ == "__main__":
    root = tk.Tk()
    trainer = TicTacToeGame(root)
    root.mainloop()
