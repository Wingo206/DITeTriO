import numpy as np
import math
import pandas as pd
import pygame
import time

class TetrisVisualizer:
    # Previous constants and piece definitions remain unchanged...
    SHAPES = {
        1: [  # I piece
            [(0,1), (1,1), (2,1), (3,1)],
            [(2,0), (2,1), (2,2), (2,3)],
            [(0,2), (1,2), (2,2), (3,2)],
            [(1,0), (1,1), (1,2), (1,3)]
        ],
        2: [  # J piece
            [(0,0), (0,1), (1,1), (2,1)],
            [(1,0), (2,0), (1,1), (1,2)],
            [(0,1), (1,1), (2,1), (2,2)],
            [(1,0), (1,1), (0,2), (1,2)]
        ],
        3: [  # L piece
            [(2,0), (0,1), (1,1), (2,1)],
            [(1,0), (1,1), (1,2), (2,2)],
            [(0,1), (1,1), (2,1), (0,2)],
            [(0,0), (1,0), (1,1), (1,2)]
        ],
        4: [  # O piece
            [(0,0), (1,0), (0,1), (1,1)],
            [(0,0), (1,0), (0,1), (1,1)],
            [(0,0), (1,0), (0,1), (1,1)],
            [(0,0), (1,0), (0,1), (1,1)]
        ],
        5: [  # S piece
            [(1,0), (2,0), (0,1), (1,1)],
            [(1,0), (1,1), (2,1), (2,2)],
            [(1,1), (2,1), (0,2), (1,2)],
            [(0,0), (0,1), (1,1), (1,2)]
        ],
        6: [  # T piece
            [(1,0), (0,1), (1,1), (2,1)],
            [(1,0), (1,1), (2,1), (1,2)],
            [(0,1), (1,1), (2,1), (1,2)],
            [(1,0), (0,1), (1,1), (1,2)]
        ],
        7: [  # Z piece
            [(0,0), (1,0), (1,1), (2,1)],
            [(2,0), (1,1), (2,1), (1,2)],
            [(0,1), (1,1), (1,2), (2,2)],
            [(1,0), (0,1), (1,1), (0,2)]
        ]
    }

    COLORS = {
        0: (0, 0, 0),      # empty - black
        1: (0, 255, 255),  # I - Cyan
        2: (0, 0, 255),    # J - Blue
        3: (255, 127, 0),  # L - Orange
        4: (255, 255, 0),  # O - Yellow
        5: (0, 255, 0),    # S - Green
        6: (128, 0, 128),  # T - Purple
        7: (255, 0, 0),    # Z - Red
        8: (200, 200, 200),    # garbage - gray
        -1: (128, 128, 128)  # Ghost/Preview - Gray
    }

    def __init__(self):
        pygame.init()
        self.CELL_SIZE = 30
        self.BOARD_WIDTH = 10
        self.BOARD_HEIGHT = 20
        self.PREVIEW_SIZE = 4
        self.MARGIN_CELLS = 4
        
        # Calculate window dimensions (added extra width for input display)
        self.window_width = (self.BOARD_WIDTH + 2 * self.MARGIN_CELLS + 12) * self.CELL_SIZE  # Increased for input display
        self.window_height = (self.BOARD_HEIGHT + 2 * self.MARGIN_CELLS) * self.CELL_SIZE
        
        # Initialize pygame window
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("Tetris Board Visualizer")
        
        # Initialize game state
        self.board = np.zeros((self.BOARD_HEIGHT, self.BOARD_WIDTH))
        self.current_piece = None
        self.hold_piece = None
        self.next_pieces = []
        self.can_hold = True
        self.inputs = np.zeros(8)  # Store input states
        self.input_change_frames = np.zeros(8)  # Store frames since last input changes

    def update_state(self, state_array):
        """Update the game state based on the 221-length input array"""
        # Update board state (indices 0-199)
        for i in range(200):
            row = i // 10
            col = i % 10
            self.board[row][col] = state_array[i]
        
        # Current piece info (200-203)
        current_piece_id = state_array[200]
        current_x = state_array[201]
        current_y = state_array[202]
        current_rotation = int(state_array[203])
        self.current_piece = {
            'id': current_piece_id,
            'x': current_x,
            'y': current_y,
            'rotation': current_rotation
        }
        
        # Hold piece info (204-205)
        self.hold_piece = state_array[204]
        self.can_hold = bool(state_array[205])
        
        # Next pieces (206-210)
        self.next_pieces = state_array[206:211]
        
        # Frames since last input change (211-218)
        self.input_change_frames = state_array[211:219]

        # Input states (219-226)
        self.inputs = state_array[219:227]

    def draw_input_display(self):
        # I'm hacky so I'm going to make input values 100-200 brightness for
        # when the input is being pressed with a probability
        """Draw the input state display"""
        # Starting position for input display
        start_x = (self.MARGIN_CELLS + self.BOARD_WIDTH + 6) * self.CELL_SIZE
        start_y = self.MARGIN_CELLS + 2 * self.CELL_SIZE
        box_size = self.CELL_SIZE * 2
        spacing = self.CELL_SIZE * 0.2
        
        # Define the layout of input boxes
        input_layout = [
            # First row
            [("180", 0, 0), ("CCW", 1, 0), ("CW", 2, 0)],
            # Second row
            [("HLD", 1, 1)],
            # Third row
            [("L", 0, 2), ("SD", 1, 2), ("R", 2, 2)],
            # Fourth row (separate)
            [("HD", 3, 0)]
        ]
        
        # Input labels to array indices mapping
        label_to_index = {
            "L": 0,    # moveleft
            "R": 1,    # moveright
            "SD": 2,   # softdrop
            "CW": 3,   # clockwise
            "CCW": 4,  # counterclockwise
            "180": 5,  # 180
            "HD": 6,   # harddrop
            "HLD": 7   # hold
        }
        
        font = pygame.font.Font(None, 36)
        
        for row in input_layout:
            for label, x_offset, y_offset in row:
                x = start_x + x_offset * (box_size + spacing)
                y = start_y + y_offset * (box_size + spacing)
                
                # Get input state if this is a monitored input
                if label in label_to_index:
                    input_state = self.inputs[label_to_index[label]]
                    # on or off
                    if input_state == 1:
                        bg_color = (255, 255, 255)
                    elif input_state == 0:
                        bg_color = (0, 0, 0)
                    else:
                        # 100 - 200
                        gray_percent = (input_state - 100) / 100.0
                        gray_percent = min(max(gray_percent, 0), 1)
                        gray_percent = math.sqrt(gray_percent)  # scale to see it better
                        gray = int(gray_percent * 255)
                        bg_color = (gray, gray, gray)
                else:
                    bg_color = (0, 0, 0)
                
                # Draw box
                pygame.draw.rect(self.screen, (0, 255, 0), 
                               (x, y, box_size, box_size), 1)
                pygame.draw.rect(self.screen, bg_color,
                               (x + 1, y + 1, box_size - 2, box_size - 2))
                
                # Draw label
                text = font.render(label, True, (255, 0, 0))
                text_rect = text.get_rect(center=(x + box_size/2, y + box_size/2))
                self.screen.blit(text, text_rect)

                # Draw frames since last change
                text2 = font.render(str(self.input_change_frames[label_to_index[label]]), True, (255, 0, 0))
                text_rect2 = text.get_rect(center=(x + box_size/2, y + 3*box_size/4))
                self.screen.blit(text2, text_rect2)

    def draw_piece(self, piece_id, x, y, rotation, color):
        """Draw a tetris piece at the specified position"""
        if piece_id < 1 or piece_id > 7:
            return
            
        shape = self.SHAPES[piece_id][rotation]
        for block_x, block_y in shape:
            pixel_x = (self.MARGIN_CELLS + x + block_x) * self.CELL_SIZE
            pixel_y = (self.MARGIN_CELLS + y + block_y) * self.CELL_SIZE
            pygame.draw.rect(self.screen, color,
                           (pixel_x, pixel_y, self.CELL_SIZE-1, self.CELL_SIZE-1))

    def draw_board(self):
        """Draw the complete board state"""
        # Clear screen
        self.screen.fill((0, 0, 0))
        
        # Draw placed pieces
        for y in range(self.BOARD_HEIGHT):
            for x in range(self.BOARD_WIDTH):
                pygame.draw.rect(self.screen, self.COLORS[int(self.board[y][x])],
                                ((self.MARGIN_CELLS + x) * self.CELL_SIZE, (self.MARGIN_CELLS + y) * self.CELL_SIZE,
                                self.CELL_SIZE-1, self.CELL_SIZE-1))
        
        # Draw current piece
        if self.current_piece:
            # account for center of piece, unless O piece
            offset = 1
            if self.current_piece['id'] == 4:
                offset = 0
            self.draw_piece(
                self.current_piece['id'],
                self.current_piece['x'] - offset,
                self.current_piece['y'] - 21,
                self.current_piece['rotation'],
                self.COLORS[self.current_piece['id']]
            )
        
        # Draw hold piece
        if self.hold_piece >= 0:
            self.draw_piece(
                self.hold_piece,
                self.BOARD_WIDTH + 1,
                1,
                0,
                self.COLORS[self.hold_piece] if self.can_hold else (128, 128, 128)
            )
        
        # Draw next pieces
        for i, piece in enumerate(self.next_pieces):
            if piece >= 0:
                self.draw_piece(
                    piece,
                    self.BOARD_WIDTH + 1,
                    i * 3 + 6,
                    0,
                    self.COLORS[piece]
                )

        # Draw gridlines
        # vertical lines
        for col in range(0, 11):
            pygame.draw.rect(self.screen, (255, 255, 255), ((self.MARGIN_CELLS + col) * self.CELL_SIZE, self.MARGIN_CELLS*self.CELL_SIZE, 1, 20 * self.CELL_SIZE))

        # horizontal lines
        for row in range(0, 21):
            pygame.draw.rect(self.screen, (255, 255, 255), (self.MARGIN_CELLS*self.CELL_SIZE, (self.MARGIN_CELLS + row) * self.CELL_SIZE, 10 * self.CELL_SIZE, 1))
        
        # Draw input display
        self.draw_input_display()
        
        pygame.display.flip()

    def run(self, state_arrays):
        """Update and display the board with the given state"""


        
        running = True
        i = 0
        while running:
            state_array = state_arrays[i]
            i += 1
            print("frame: " + str(i))

            self.update_state(state_array)
            self.draw_board()

            time.sleep(0.1)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
        
        pygame.quit()

# Example usage
if __name__ == "__main__":
    # # Create a sample state array (221 elements)
    # sample_state = np.zeros(221)

    # # sample values in the board
    # sample_state[150:158] = 1
    # sample_state[158:166] = 3
    # sample_state[190:200] = 8
    
    # # Set some example values
    # # Put an I piece (id=4) in the middle of the board
    # sample_state[200] = 1  # Current piece is I
    # sample_state[201] = 3  # x position
    # sample_state[202] = 10  # y position
    # sample_state[203] = 0  # rotation
    
    # # Set hold piece
    # sample_state[204] = 2  # O piece in hold
    # sample_state[205] = 1  # Can hold
    
    # # Set next pieces
    # sample_state[206:211] = [2, 4, 6, 1, 7]  # Next 5 pieces
    
    # # Set some example input states
    # sample_state[212:220] = [1, 0, 1, 0, 0, 1, 0, 1]  # Some random input states


    # Read list of states from a file
    df = pd.read_csv("data/processed_replays/test2_0.csv", index_col=False)
    # df = pd.read_csv("data/processed_replays/players/sodiumoverdose/6657e2e7cdcf03ad6260a6d8_p1_r0.csv", index_col=False)
    print(len(df.iloc[0]))
    print(df.iloc[0][200:])
    print(df.iloc[10][211:219])
    print(df.iloc[10][219:227])

    states = df.to_numpy()
    print(states[0][204:])
    
    # Create and run visualizer
    visualizer = TetrisVisualizer()
    visualizer.run(states)
