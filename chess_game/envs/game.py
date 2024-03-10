from typing import List, Tuple, Optional
from string import ascii_uppercase as alphabet
import re
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from random import randrange, shuffle
from itertools import product, combinations
from math import sqrt, ceil, isclose

from envs.environment import AbstractPlayer, AbstractState, AbstractAction

import chess


# Interface to the chess package for Python
#
# WHITE is Player 0
# BLACK is Player 1

class ID(Enum):
    """ Player ID """

@dataclass
class Player(AbstractPlayer):
    name: str                        # Player name
    id_: ID                          # Player id

@dataclass
class Action(AbstractAction):
    chessmove : chess.Move             # Move as in chess/Python

    def __init__(self,move):
        self.chessmove = move

    def __str__(self):
        return self.chessmove.uci()

    def __eq__(self,other):
        return self.chessmove == other.chessmove


FIRSTROW = [chess.Piece(chess.KNIGHT,chess.WHITE),
            chess.Piece(chess.QUEEN,chess.WHITE),
            chess.Piece(chess.KING,chess.WHITE),
            chess.Piece(chess.BISHOP,chess.WHITE),
            chess.Piece(chess.KNIGHT,chess.WHITE)]
LASTROW = [chess.Piece(chess.KNIGHT,chess.BLACK),
           chess.Piece(chess.BISHOP,chess.BLACK),
           chess.Piece(chess.KING,chess.BLACK),
           chess.Piece(chess.QUEEN,chess.BLACK),
           chess.Piece(chess.KNIGHT,chess.BLACK)]

class State(AbstractState):

    # Initialize the Chess board
    #
    # maxX = index of last file (column), when first column is 0
    # maxY = index of last rank (row), when first row is 0

    def __init__(self,players,maxX=4,maxY=4):
        self.__players = players
        self.board = chess.Board()
        self.board.clear_board()

        for i,p in list(enumerate(FIRSTROW)):
            self.board.set_piece_at(chess.parse_square(chess.FILE_NAMES[i] + "1"),p)

        for i,p in list(enumerate(LASTROW)):
            self.board.set_piece_at(chess.parse_square(chess.FILE_NAMES[i] + str(maxY+1)),p)

        self.maxX = maxX
        self.maxY = maxY

    def current_player(self) -> int:
        if self.board.turn == chess.WHITE:
            return 0
        else:
            return 1

    def current_player_id(self) -> ID:
        return self.__players[self.current_player()].id_
    
    def __str__(self) -> str:
        # Get whole board in Unicode
        s = self.board.unicode()
        # Split it to rows/ranks
        ranks = s.split("\n")
        visibleRanks = list(enumerate(ranks[7-self.maxY:]))
        # Return maxX prefixes of maxY rows/ranks
        return '\n'.join([ "  abcdefgh"[:(self.maxX)*2]] + [ str(self.maxX+2-i) + " " + s[:(self.maxX+1)*2] for i,s in visibleRanks ])


    # Is somebody the winner? 1 for current player, -1 for opponent

    def is_winner(self) -> Optional[int]:
        outc = self.board.outcome()
        if outc == None:
            return None
        if outc.winner == chess.WHITE and self.board.turn == chess.WHITE:
            return 1
        else:
            return -1


    # Test if move is within bounds 0..maxX, 0..maxY

    def withinBounds(self,m):
        src = m.from_square
        dest = m.to_square
#        print("Move: " +str(self.board.piece_at(m.from_square)) + " " + chess.square_name(src) + " to " +chess.square_name(dest))
        return chess.square_file(dest) <= self.maxX and chess.square_rank(dest) <= self.maxY

    # Applicable moves in the current state

    def applicable_moves(self) -> List[Action]:
        return [ Action(m) for m in list(self.board.legal_moves) if self.withinBounds(m) ]
        
    # Perform action in the current state

    def execute_move(self,m):
        self.board.push(m.chessmove)

    # Undo changes caused by last action, to enable backtracking

    def undo_last_move(self):
        self.board.pop()

    def clone(self):
        newstate = State(self.__players)
        newstate.board = self.board.copy()
        newstate.maxX = self.maxX
        newstate.maxY = self.maxY
        return newstate
