import random
import sys

def display_board(board):
    def cell(val, idx):
        return val if val != ' ' else str(idx + 1)

    print(f" {cell(board[0],0)} | {cell(board[1],1)} | {cell(board[2],2)} ")
    print("---+---+---")
    print(f" {cell(board[3],3)} | {cell(board[4],4)} | {cell(board[5],5)} ")
    print("---+---+---")
    print(f" {cell(board[6],6)} | {cell(board[7],7)} | {cell(board[8],8)} ")
    print()


def check_win(board, symbol):
    wins = [
        (0,1,2), (3,4,5), (6,7,8),  # rows
        (0,3,6), (1,4,7), (2,5,8),  # cols
        (0,4,8), (2,4,6)            # diagonals
    ]
    return any(all(board[i] == symbol for i in combo) for combo in wins)


def choose_game_mode():
    while True:
        print("Select game mode:")
        print("1. human vs human")
        print("2. human vs computer")
        choice = input("Enter 1 or 2: ").strip()
        if choice in ('1', '2'):
            return int(choice)
        print("Invalid choice. Please enter 1 or 2.")


def get_player_info(mode):
    players = []
    available = ['X', 'O']

    if mode == 1:
        # Two human players
        for i in (1, 2):
            name = input(f"Enter name for Player {i}: ").strip() or f"Player{i}"
            # Symbol selection
            while True:
                choice = input(f"{name}, choose your symbol (X/O) or press Enter for random: ").strip().upper()
                if choice == '':
                    sym = random.choice(available)
                elif choice in available:
                    sym = choice
                else:
                    print("Invalid or unavailable symbol, assigning randomly.")
                    sym = random.choice(available)
                available.remove(sym)
                players.append({'name': name, 'symbol': sym, 'is_computer': False})
                break

    else:
        # Human vs Computer
        human = input("Enter your name: ").strip() or "Human"
        # Human symbol
        while True:
            choice = input(f"{human}, choose your symbol (X/O) or press Enter for random: ").strip().upper()
            if choice == '':
                sym = random.choice(available)
            elif choice in available:
                sym = choice
            else:
                print("Invalid or unavailable symbol, assigning randomly.")
                sym = random.choice(available)
            available.remove(sym)
            players.append({'name': human, 'symbol': sym, 'is_computer': False})
            break
        # Computer player
        comp_sym = available[0]
        players.append({'name': 'Computer', 'symbol': comp_sym, 'is_computer': True})

    return players


def get_human_move(board, player):
    while True:
        try:
            move = int(input(f"{player['name']} ({player['symbol']}), choose your move (1-9): "))
            if move < 1 or move > 9:
                print("Please enter a number between 1 and 9.")
                continue
            if board[move - 1] != ' ':
                print("That spot is already taken. Try again.")
                continue
            return move - 1
        except ValueError:
            print("Invalid input. Enter a number between 1 and 9.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")


def get_computer_move(board, player):
    available_moves = [i for i, spot in enumerate(board) if spot == ' ']
    move = random.choice(available_moves)
    print(f"Computer ({player['symbol']}) chooses position {move+1}")
    return move


def play_game():
    mode = choose_game_mode()
    players = get_player_info(mode)
    board = [' '] * 9
    current = 0  # index of current player

    while True:
        display_board(board)
        player = players[current]
        try:
            if player['is_computer']:
                move = get_computer_move(board, player)
            else:
                move = get_human_move(board, player)
        except Exception:
            print("Error during move selection. Skipping turn.")
            current = 1 - current
            continue

        board[move] = player['symbol']

        # Check for win
        if check_win(board, player['symbol']):
            display_board(board)
            print(f"\nCongratulations {player['name']}! You have won!\n")
            return

        # Check for tie
        if ' ' not in board:
            display_board(board)
            print("\nThe game is a tie!\n")
            return

        # Switch turns
        current = 1 - current


def main():
    print("Welcome to Tic-Tac-Toe!\n")
    while True:
        try:
            play_game()
        except Exception as e:
            print(f"An unexpected error occurred: {e}\nStarting a new game...\n")
            continue

        again = input("Do you want to play again? (y/n): ").strip().lower()
        if not again.startswith('y'):
            print("\nThank you for playing! Goodbye.")
            break


if __name__ == '__main__':
    main()
