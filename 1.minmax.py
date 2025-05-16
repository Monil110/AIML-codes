def printBoard():
    print(board[1], '|', board[2], '|', board[3])
    print(board[4], '|', board[5], '|', board[6])
    print(board[7], '|', board[8], '|', board[9])


def spaceIsFree(pos):
    return board[pos] == ' '


def insertLetter(letter, pos):
    if spaceIsFree(pos):
        board[pos] = letter
        printBoard()
        if checkDraw():
            print("Draw!")
            exit()
        if checkForWin():
            print("Bot wins!" if letter == 'X' else "Player wins!")
            exit()
    else:
        pos = int(input("Can't insert there! Enter new position: "))
        insertLetter(letter, pos)


def checkForWin():
    win = [[1,2,3],[4,5,6],[7,8,9],[1,4,7],[2,5,8],[3,6,9],[1,5,9],[7,5,3]]
    for combo in win:
        if board[combo[0]] == board[combo[1]] == board[combo[2]] != ' ':
            return True
    return False


def checkWhichMarkWon(mark):
    win = [[1,2,3],[4,5,6],[7,8,9],[1,4,7],[2,5,8],[3,6,9],[1,5,9],[7,5,3]]
    for combo in win:
        if board[combo[0]] == board[combo[1]] == board[combo[2]] == mark:
            return True
    return False


def checkDraw():
    return all(board[key] != ' ' for key in board)


def playerMove():
    pos = int(input("Enter the position for 'O': "))
    insertLetter(player, pos)


def compMove():
    bestScore = -800
    move = None
    for key in board:
        if board[key] == ' ':
            board[key] = bot
            score = minimax(board, False)
            board[key] = ' '
            if score > bestScore:
                bestScore = score
                move = key
    if move:
        insertLetter(bot, move)
    else:
        print("No valid moves left. It's a draw!")
        exit()


def minimax(board, isMax):
    if checkWhichMarkWon(bot):
        return 1
    if checkWhichMarkWon(player):
        return -1
    if checkDraw():
        return 0

    if isMax:
        best = -800
        for key in board:
            if board[key] == ' ':
                board[key] = bot
                score = minimax(board, False)
                board[key] = ' '
                best = max(score, best)
        return best
    else:
        best = 800
        for key in board:
            if board[key] == ' ':
                board[key] = player
                score = minimax(board, True)
                board[key] = ' '
                best = min(score, best)
        return best


board = {i: ' ' for i in range(1, 10)}
printBoard()
print("Player goes first! Good luck.")
print("Positions are as follow:")
print("1, 2, 3")
print("4, 5, 6")
print("7, 8, 9\n")
player = 'O'
bot = 'X'

while not checkForWin():
    playerMove()
    if checkForWin():
        break
    compMove()
